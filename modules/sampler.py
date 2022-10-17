from .__common__ import *
from utils import sphere
from utils.torch_ext import grid2d

__all__ = ["Sampler", "UniformSampler", "PdfSampler"]


class Sampler(nn.Module):
    _samples_indices_cached: torch.Tensor | None

    def __init__(self, x_chns: int, d_chns: int):
        """
        Initialize a Sampler module
        """
        super().__init__({}, {"x": x_chns, "d": d_chns})
        self._samples_indices_cached = None

    def __call__(self, rays: Rays, **kwargs) -> Samples:
        raise NotImplementedError

    def _get_samples_indices(self, pts: torch.Tensor) -> torch.Tensor:
        """
        Get 2D indices of samples. The first value is the index of ray, while the second value is 
        the index of sample in a ray.

        :param pts `Tensor(B, P, 3)`: the sample points
        :return `Tensor(B, P)`: the 2D indices of samples
        """
        if self._samples_indices_cached is None\
                or self._samples_indices_cached.device != pts.device\
                or self._samples_indices_cached.shape[0] < pts.shape[0]\
                or self._samples_indices_cached.shape[1] < pts.shape[1]:
            self._samples_indices_cached = grid2d(*pts.shape[:2], indexing="ij", device=pts.device)
        return self._samples_indices_cached[:pts.shape[0], :pts.shape[1]]

    def _get_samples(self, rays: Rays, t_vals: torch.Tensor, mode: str) -> Samples:
        """
        Get samples along rays at sample steps specified by `t_vals`.

        :param rays_o `Tensor(B, 3)`: rays' origin
        :param rays_d `Tensor(B, 3)`: rays' direction
        :param t_vals `Tensor(B, P)`: sample steps
        :param mode `str`: sample mode, one of "xyz", "xyz_disp", "spherical", "spherical_radius"
        :return `Samples(B, P)`: samples
        """
        if mode == "xyz":
            z_vals = t_vals
            pts = rays.get_points(z_vals)
        elif mode == "xyz_disp":
            z_vals = t_vals.reciprocal()
            pts = rays.get_points(z_vals)
        elif mode == "spherical":
            z_vals = t_vals.reciprocal()
            pts = sphere.cartesian2spherical(rays.get_points(z_vals), inverse_r=True)
        elif mode == "spherical_radius":
            z_vals = sphere.ray_sphere_intersect(rays, t_vals.reciprocal())
            pts = sphere.cartesian2spherical(rays.get_points(z_vals), inverse_r=True)
        else:
            raise ValueError(f"Unknown mode: {mode}")

        rays_d = rays.rays_d.unsqueeze(1)  # (B, 1, 3)
        dists = union(z_vals[..., 1:] - z_vals[..., :-1], math.huge)  # (B, P)
        dists *= torch.norm(rays_d, dim=-1)
        return Samples(
            pts=pts,
            dirs=rays_d.expand(*pts.shape[:2], -1),
            depths=z_vals,
            t_vals=t_vals,
            dists=dists,
            indices=self._get_samples_indices(pts)
        )


class UniformSampler(Sampler):
    """
    This module expands NeRF's code of uniform sampling to support our spherical sampling and enable
    the trace of samples' indices.
    """

    def __init__(self):
        super().__init__(3, 3)

    def _sample(self, range: tuple[float, float], n_rays: int, n_samples: int, perturb: bool) -> torch.Tensor:
        """
        Generate sample steps along rays in the specified range.

        :param range `float, float`: sampling range
        :param n_rays `int`: number of rays (B)
        :param n_samples `int`: number of samples per ray (P)
        :param perturb `bool`: whether perturb sampling
        :return `Tensor(B, P)`: sampled "t"s along rays
        """
        t_vals = torch.linspace(*range, n_samples, device=self.device)  # (P)
        if perturb:
            mids = .5 * (t_vals[..., 1:] + t_vals[..., :-1])
            upper = union(mids, t_vals[..., -1:])
            lower = union(t_vals[..., :1], mids)
            # stratified samples in those intervals
            t_vals = t_vals.expand(n_rays, -1)
            t_vals = lower + (upper - lower) * torch.rand_like(t_vals)
        else:
            t_vals = t_vals.expand(n_rays, -1)
        return t_vals

    def __call__(self, rays: Rays, *,
                 range: tuple[float, float],
                 mode: str,
                 n_samples: int,
                 perturb: bool) -> Samples:
        """
        Sample points along rays.

        :param rays `Rays(B)`: rays
        :param range `float, float`: sampling range
        :param mode `str`: sample mode, one of "xyz", "xyz_disp", "spherical", "spherical_radius"
        :param n_samples `int`: number of samples per ray
        :param perturb `bool`: whether perturb sampling, defaults to `False`
        :return `Samples(B, P)`: samples
        """
        t_range = range if mode == "xyz" else (1. / range[0], 1. / range[1])
        t_vals = self._sample(t_range, rays.shape[0], n_samples, perturb)  # (B, P)
        return self._get_samples(rays, t_vals, mode)


class PdfSampler(Sampler):
    """
    Hierarchical sampling (section 5.2 of NeRF)
    """

    def __init__(self):
        super().__init__(3, 3)

    def _sample(self, t_vals: torch.Tensor, weights: torch.Tensor, n_importance: int,
                perturb: bool, include_existed: bool, sort_descending: bool) -> torch.Tensor:
        """
        Generate sample steps by PDF according to existed sample steps and their weights.

        :param t_vals `Tensor(B, P)`: existed sample steps
        :param weights `Tensor(B, P)`: weights of existed sample steps
        :param n_importance `int`: number of samples to generate for each ray
        :param perturb `bool`: whether perturb sampling
        :param include_existed `bool`: whether to include existed samples in the output
        :return `Tensor(B, P'[+P])`: the output sample steps
        """
        bins = .5 * (t_vals[..., 1:] + t_vals[..., :-1])  # (B, P - 1)
        weights = weights[..., 1:-1] + math.tiny  # (B, P - 2)

        # Get PDF
        pdf = weights / torch.sum(weights, -1, keepdim=True)
        cdf = union(0., torch.cumsum(pdf, -1))  # (B, P - 1)

        # Take uniform samples
        if perturb:
            u = torch.rand(*cdf.shape[:-1], n_importance, device=self.device)
        else:
            u = torch.linspace(0., 1., steps=n_importance, device=self.device).\
                expand(*cdf.shape[:-1], -1)

        # Invert CDF
        u = u.contiguous()  # (B, P')
        inds = torch.searchsorted(cdf, u, right=True)  # (B, P')
        inds_g = torch.stack([
            (inds - 1).clamp_min(0),  # below
            inds.clamp_max(cdf.shape[-1] - 1)  # above
        ], -1)  # (B, P', 2)

        matched_shape = [inds_g.shape[0], inds_g.shape[1], cdf.shape[-1]]  # [B, P', P - 1]
        cdf_g = torch.gather(cdf.unsqueeze(1).expand(matched_shape), 2, inds_g)  # (B, P', 2)
        bins_g = torch.gather(bins.unsqueeze(1).expand(matched_shape), 2, inds_g)  # (B, P', 2)

        denom = cdf_g[..., 1] - cdf_g[..., 0]
        denom = torch.where(denom < math.tiny, torch.ones_like(denom), denom)
        t = (u - cdf_g[..., 0]) / denom
        t_samples = (bins_g[..., 0] + t * (bins_g[..., 1] - bins_g[..., 0])).detach()
        if include_existed:
            return torch.sort(union(t_vals, t_samples), -1, descending=sort_descending)[0]
        else:
            return t_samples

    def __call__(self, rays: Rays, t_vals: torch.Tensor, weights: torch.Tensor, *,
                 mode: str,
                 n_importance: int,
                 perturb: bool,
                 include_existed_samples: bool) -> Samples:
        """
        Sample points along rays using PDF sampling based on existed samples.

        :param rays `Rays(B)`: rays
        :param t_vals `Tensor(B, P)`: existed sample steps
        :param weights `Tensor(B, P)`: weights of existed sample steps
        :param mode `str`: sample mode, one of "xyz", "xyz_disp", "spherical", "spherical_radius"
        :param n_importance `int`: number of samples to generate using PDF sampling for each ray
        :param perturb `bool`: whether perturb sampling, defaults to `False`
        :param include_existed_samples `bool`: whether to include existed samples in the output,
            defaults to `True`
        :return `Samples(B, P'[+P])`: samples
        """
        t_vals = self._sample(t_vals, weights, n_importance, perturb, include_existed_samples,
                              mode != "xyz")
        return self._get_samples(rays, t_vals, mode)
