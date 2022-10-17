import torch
from . import math
from .types import *


def cartesian2spherical(cart: Tensor, inverse_r: bool = False) -> Tensor:
    """
    Convert coordinates from Cartesian to Spherical

    :param cart `Tensor([N...,] 3)`: coordinates in Cartesian
    :param inverse_r: whether to convert r to reciprocal form, defaults to `False`
    :return `Tensor([N...,] 3)`: coordinates in Spherical ([r | 1/r], theta, phi)
    """
    rho = torch.sqrt(torch.sum(cart * cart, dim=-1))
    theta = math.get_angle(cart[..., 2], cart[..., 0])
    if inverse_r:
        rho = rho.reciprocal()
        phi = torch.asin(cart[..., 1] * rho)
    else:
        phi = torch.asin(cart[..., 1] / rho)
    return torch.stack([rho, theta, phi], dim=-1)


def spherical2cartesian(spher: Tensor, inverse_r: bool = False) -> Tensor:
    """
    Convert coordinates from Spherical to Cartesian

    :param spher `Tensor([N...,] 3)`: coordinates in Spherical  ([r | 1/r], theta, phi)
    :param inverse_r `bool`: whether r is in reciprocal form, defaults to `False`
    :return `Tensor([N...,] 3)`:, coordinates in Cartesian
    """
    rho = spher[..., 0]
    if inverse_r:
        rho = rho.reciprocal()
    sin_theta_phi = torch.sin(spher[..., 1:3])
    cos_theta_phi = torch.cos(spher[..., 1:3])
    x = rho * sin_theta_phi[..., 0] * cos_theta_phi[..., 1]
    y = rho * sin_theta_phi[..., 1]
    z = rho * cos_theta_phi[..., 0] * cos_theta_phi[..., 1]
    return torch.stack([x, y, z], dim=-1)


def ray_sphere_intersect(rays: Rays, r: Tensor) -> Tensor:
    """
    Calculate intersections of each rays and each spheres

    :param rays `Rays(B)`: rays
    :param r `Tensor(P)`: , radius of spheres
    :return `Tensor(B, P)`: depths of intersections along rays
    """
    # p, v: Expand to (B, 1, 3)
    p = rays.rays_o.unsqueeze(1)
    v = rays.rays_d.unsqueeze(1)
    # pp, vv, pv: (B, 1)
    pp = (p * p).sum(dim=2)
    vv = (v * v).sum(dim=2)
    pv = (p * v).sum(dim=2)
    z = (((pv * pv - vv * (pp - r * r)).sqrt() - pv) / vv)  # (B, P)
    return z


def get_rot_matrix(theta: float | Tensor, phi: float | Tensor) -> Tensor:
    """
    Get rotation matrix from angles in spherical space

    :param theta `Tensor([N...,] 1) | float`: rotation angles around y axis
    :param phi  `Tensor([N...,] 1) | float`: rotation angles around x axis
    :return: `Tensor([N...,] 3, 3)` rotation matrices
    """
    if not isinstance(theta, Tensor):
        theta = torch.scalar_tensor(theta)
    if not isinstance(phi, Tensor):
        phi = torch.scalar_tensor(phi)
    spher = torch.cat([torch.ones_like(theta), theta, phi], dim=-1)
    forward = spherical2cartesian(spher)  # ([N...,] 3)
    up = Tensor([0., 1., 0.])
    forward, up = torch.broadcast_tensors(forward, up)
    right = torch.cross(forward, up, dim=-1)  # ([N...,] 3)
    up = torch.cross(right, forward, dim=-1)  # ([N...,] 3)
    return torch.stack([right, up, forward], dim=-2)  # ([N...,] 3, 3)
