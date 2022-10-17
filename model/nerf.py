from .__common__ import *
from .model import Model


class NeRF(Model):
    class Args(Model.Args):
        n_samples: int = 64
        sample_mode: str = "xyz"
        perturb_sampling: bool = False
        depth: int = 8
        width: int = 256
        skips: list[int] = [4]
        act: str = "relu"
        color_decoder: str = "NeRF"
        n_importance: int = 0
        fine_depth: int = 8
        fine_width: int = 256
        fine_skips: list[int] = [4]
        xfreqs: int = 10
        dfreqs: int = 4
        raw_noise_std: float = 0.
        near: float = 1.
        far: float = 10.

    args: Args

    def __init__(self, args: Args):
        """
        Initialize a NeRF model

        :param args `dict`: arguments
        """
        super().__init__(args)
        if args.sample_mode == "xyz" or args.sample_mode == "xyz_disp":
            args.near = 0.1

        # Initialize components
        self._init_sampler()
        self._init_encoders()
        self._init_core()
        self._init_renderer()

        if self.args.n_importance > 0:
            self._init_cascade()

    def __call__(self, rays: Rays, *outputs: str, **args) -> RenderOut:
        samples = self.sample(rays, **args)
        x, d = self.encode(samples)
        rgbd = self.infer(x, d)
        return self.render(rays, samples, rgbd, *outputs, cascade=True, **args)

    def sample(self, rays: Rays, **kwargs) -> Samples:
        args = self.args.merge_with(kwargs)
        return self.sampler(rays, range=(args.near, args.far),
                            mode=args.sample_mode, n_samples=args.n_samples,
                            perturb=args.perturb_sampling if self.training else False)

    def encode(self, samples: Samples) -> tuple[Tensor, Tensor]:
        return self.x_encoder(samples.pts), self.d_encoder(math.normalize(samples.dirs))

    def infer(self, x: Tensor, d: Tensor, *, fine: bool = False) -> Tensor:
        if self.args.n_importance > 0 and fine:
            return self.fine_core(x, d)
        return self.core(x, d)

    def render(self, rays: Rays, samples: Samples, rgbd: Tensor, *outputs: str,
               cascade: bool = False, **kwargs) -> RenderOut:
        args = self.args.merge_with(kwargs)
        if args.n_importance > 0 and cascade:
            coarse_outputs = [item[7:] for item in outputs if item.startswith("coarse_")]
            coarse_ret = self.renderer(samples, rgbd, "weights", *coarse_outputs,
                                       white_bg=args.white_bg,
                                       raw_noise_std=args.raw_noise_std if self.training else 0.)
            samples = self.pdf_sampler(rays, samples.t_vals, coarse_ret["weights"][..., 0],
                                       mode=args.sample_mode,
                                       n_importance=args.n_importance,
                                       perturb=args.perturb_sampling if self.training else False,
                                       include_existed_samples=True)
            x, d = self.encode(samples)
            fine_rgbd = self.infer(x, d, fine=True)
            return self.renderer(samples, fine_rgbd, *outputs, white_bg=args.white_bg,
                                 raw_noise_std=args.raw_noise_std if self.training else 0.) | {
                f"coarse_{key}": coarse_ret[key]
                for key in coarse_outputs
                if key in coarse_ret
            }
        return self.renderer(samples, rgbd, *outputs, white_bg=False,
                             raw_noise_std=args.raw_noise_std if self.training else 0.)

    def _init_encoders(self):
        self.x_encoder = FreqEncoder(self.sampler.out_chns["x"], self.args.xfreqs, True)
        self.d_encoder = FreqEncoder(self.sampler.out_chns["d"], self.args.dfreqs, True)

    def _init_core(self):
        self.core = core.NeRF(self.x_encoder.out_chns, self.d_encoder.out_chns, 3,
                              self.args.depth, self.args.width, self.args.skips,
                              self.args.act, self.args.color_decoder)

    def _init_sampler(self):
        self.sampler = UniformSampler()

    def _init_cascade(self):
        self.pdf_sampler = PdfSampler()
        self.fine_core = core.NeRF(self.x_encoder.out_chns, self.d_encoder.out_chns, 3,
                                   self.args.fine_depth, self.args.fine_width, self.args.fine_skips,
                                   self.args.act, self.args.color_decoder)

    def _init_renderer(self):
        self.renderer = VolumnRenderer()
