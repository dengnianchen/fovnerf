from .__common__ import *
from .model import Model


class FovNeRF(Model):

    class Args(Model.Args):
        n_samples: int = 64
        perturb_sampling: bool = False
        with_radius: bool = False
        n_fields: int = 1
        depth: int = 8
        width: int = 256
        skips: list[int] = [4]
        act: str = "relu"
        xfreqs: int = 6
        raw_noise_std: float = 0.
        near: float = 1.
        far: float = 10.

    args: Args

    def __init__(self, args: Args):
        """
        Initialize a FS-NeRF model

        :param args `Args`: arguments
        """
        super().__init__(args)

        # Initialize components
        self._init_sampler()
        self._init_encoders()
        self._init_core()
        self._init_renderer()

    def __call__(self, rays: Rays, *outputs: str, **args) -> RenderOut:
        samples = self.sample(rays, **args)
        x = self.encode(samples)
        rgbd = self.infer(x)
        return self.render(samples, rgbd, *outputs, **args)

    def export(self, out_dir: Path, name: str, batch_size: int) -> list[Path]:
        Model.export_onnx(self.core, out_dir / f"{name}.onnx",
                          {
                              'Encoded': [batch_size, self.args.n_samples, self.x_encoder.out_chns],
                          },
                          ['RGBD'])
        with open(out_dir / f"{name}.ini", "w") as fp:
            fp.write(f"model={self.__class__.__name__}\n")
            for param_name in ["n_samples", "with_radius", "xfreqs", "near", "far"]:
                fp.write(f"{param_name}={getattr(self.args, param_name)}\n")
        return [out_dir / f"{name}.onnx"]

    def sample(self, rays: Rays, **kwargs) -> Samples:
        args = self.args.merge_with(kwargs)
        return self.sampler(rays, range=(args.near, args.far), mode="spherical_radius",
                            n_samples=args.n_samples,
                            perturb=args.perturb_sampling if self.training else False)

    def encode(self, samples: Samples) -> Tensor:
        return self.x_encoder(samples.pts[..., -self.x_encoder.in_chns:])

    def infer(self, x: Tensor) -> Tensor:
        return self.core(x)

    def render(self, samples: Samples, rgbd: Tensor, *outputs: str, **kwargs) -> RenderOut:
        args = self.args.merge_with(kwargs)
        return self.renderer(samples, rgbd, *outputs, white_bg=False,
                             raw_noise_std=args.raw_noise_std if self.training else 0.)

    def _init_encoders(self):
        self.x_encoder = FreqEncoder(self.sampler.out_chns["x"] - (not self.args.with_radius),
                                     self.args.xfreqs, False)

    def _init_core(self):
        self.core = core.FovNeRF(self.x_encoder.out_chns, 3, self.args.depth, self.args.width,
                                 self.args.skips, self.args.act, self.args.n_samples,
                                 self.args.n_fields)

    def _init_sampler(self):
        self.sampler = UniformSampler()

    def _init_renderer(self):
        self.renderer = VolumnRenderer()
