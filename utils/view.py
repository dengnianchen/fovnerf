import torch

from .types import *
from .math import fov2length
from .torch_ext import union, grid2d

__all__ = ["Camera", "Trans"]


class Camera(object):
    res: Resolution
    f: Tensor
    c: Tensor
    forward: float
    device: torch.device
    _local_rays_cached: Tensor

    @property
    def local_rays(self) -> Tensor:
        if self._local_rays_cached is None:
            self._build_local_rays()
        return self._local_rays_cached

    def __init__(self, params: dict[str, Any], res: Resolution, device: torch.device = None):
        super().__init__()
        self.res = res
        self.device = device
        self.forward = -1.
        self._local_rays_cached = None
        params = Camera.convert_params(params, self.res)
        self.f = torch.tensor(params["f"], device=self.device, dtype=torch.float32)
        self.c = torch.tensor(params["c"], device=self.device, dtype=torch.float32)

    def to(self, device: torch.device):
        self.device = device
        self.f = self.f.to(self.device)
        self.c = self.c.to(self.device)
        self._local_rays_cached = None if self._local_rays_cached is None else\
            self._local_rays_cached.to(self.device)
        return self

    def resize(self, res: Resolution):
        old_res = self.res
        new_res = res
        self._resize(old_res, new_res)
        self.res = res
        self._local_rays_cached = None

    def get_pixels(self, image: Tensor) -> Tensor:
        return image.movedim(-3, -1).flatten(-3, -2)

    def proj(self, p: Tensor, normalize: bool = False, center_as_origin: bool = False) -> Tensor:
        """
        Project positions in camera space to image plane

        :param p `Tensor(..., 3)`: positions in local space
        :param normalize: use normalized coord for image plane
        :param center_as_origin: take center as the origin if image plane instead of top-left corner
        :return `Tensor(..., 2)`: positions in image plane
        """
        p = p[..., :2] / p[..., 2:] * self.forward * self.f
        if not center_as_origin:
            p = p + self.c
        if normalize:
            p = p / torch.tensor([self.res.w - 1, self.res.h - 1], device=self.device)
        return p

    def unproj(self, p: Tensor) -> Tensor:
        """
        Unproject positions in image plane to camera space

        :param p `Tensor(..., 2)`: positions in image plane
        :return: positions in local space
        """
        return union((p - self.c) / self.f, self.forward)

    def get_local_rays(self, normalize: bool = False, flatten: bool = False) -> Tensor:
        """
        Get view rays in camera space

        :param normalize: whether normalize rays to unit length, defaults to False
        :param flatten: whether flatten the return tensor, defaults to False
        :return `Tensor(H, W, 3)|Tensor(HW, 3)`: the shape is determined by parameter 'flatten'
        """
        pixels = grid2d(*self.res, device=self.device)
        rays = self.unproj(pixels)
        if normalize:
            rays /= rays.norm(dim=-1, keepdim=True)
        if flatten:
            rays = rays.flatten(0, 1)
        return rays

    @staticmethod
    def convert_params(input_params: dict[str, Any], res: Resolution) -> dict[str, Any]:
        """
        Check and convert camera parameters in config file to pixel-space

        :param cam_params `{str: any}`: the parameters of camera,
            { [("f": float | [float, float]) | ("fov": float)], "c": float | [float, float], ["normalized": bool] },
        :param res `Resolution`: resolution of view
        :return `{str: any}`: converted camera parameters, {"f": [float, float], "c": [float, float]}
        """
        input_is_normalized = input_params.get("normalized", False)
        params = {}
        if "fov" in input_params:
            params["f"] = [res.h / fov2length(input_params["fov"])] * 2
            params["f"][1] *= -1
        else:
            params["f"] = input_params["f"] if isinstance(input_params["f"], list)\
                else [input_params["f"], -input_params["f"]]
            if input_is_normalized:
                params["f"][0] *= res.w
                params["f"][1] *= res.h

        if "c" not in input_params:
            params["c"] = [res.w / 2, res.h / 2]
        else:
            params["c"] = input_params["c"] if isinstance(input_params["c"], list)\
                else [input_params["c"]] * 2
            if input_is_normalized:
                params["c"][0] *= res.w
                params["c"][1] *= res.h
        return params

    def _resize(self, old_res: Resolution, new_res: Resolution):
        scale = torch.tensor([new_res.w / old_res.w, new_res.h / old_res.h], device=self.device)
        self.f.mul_(scale)
        self.c.mul_(scale)

    def _build_local_rays(self):
        self._local_rays_cached = self.get_local_rays(flatten=True)


class Trans(object):

    @property
    def shape(self) -> torch.Size:
        return self.t.shape[:-1]

    def __init__(self, t: Tensor, r: Tensor):
        self.t = t
        self.r = r
        if len(self.t.size()) == 1:
            self.t = self.t[None, :]
            self.r = self.r[None, :, :]
        self.to(self.t.device)

    def to(self, device: torch.device) -> "Trans":
        self.t = self.t.to(device)
        self.r = self.r.to(device)
        self.device = device
        return self

    def trans_point(self, p: Tensor, inverse=False) -> Tensor:
        """
        Transform points by given translation vectors and rotation matrices

        :param p `Tensor(N.., 3)`: points to transform
        :param inverse: whether perform inverse transform
        :return `Tensor(M.., N.., 3)`: transformed points
        """
        size_N = list(p.size())[:-1]
        size_M = list(self.r.size())[:-2]
        out_size = size_M + size_N + [3]
        t_size = size_M + [1 for _ in range(len(size_N))] + [3]
        t = self.t.view(t_size)  # (M.., 1.., 3)
        if inverse:
            p = (p - t).view(size_M + [-1, 3])
            r = self.r
        else:
            p = p.view(-1, 3)
            r = self.r.movedim(-1, -2)  # Transpose rotation matrices
        out = torch.matmul(p, r).view(out_size)
        if not inverse:
            out = out + t
        return out

    def trans_vector(self, v: Tensor, inverse=False) -> Tensor:
        """
        Transform vectors by given translation vectors and rotation matrices

        :param v `Tensor(N.., 3)`: vectors to transform
        :param inverse: whether perform inverse transform
        :return `Tensor(M.., N.., 3)`: transformed vectors
        """
        out_size = list(self.r.size())[:-2] + list(v.size())[:-1] + [3]
        r = self.r.movedim(-1, -2) if inverse else self.r
        return (r.unsqueeze(-3) @ v.reshape(-1, 3, 1)).reshape(out_size)

    def reshape(self, *size) -> "Trans":
        return Trans(self.t.reshape(*size, 3), self.r.reshape(*size, 3, 3))

    def __getitem__(self, index: IndexSelector) -> "Trans":
        return Trans(self.t[index], self.r[index])
