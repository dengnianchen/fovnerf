import numpy as np
import torch
import glm
from typing import TypeVar
from math import *

Value = TypeVar("Value", float, torch.Tensor, np.ndarray)
Tensor = TypeVar("Tensor", torch.Tensor, np.ndarray)

huge = 1e10
tiny = 1e-7


def sin(x: Value) -> Value:
    return torch.sin(x) if isinstance(x, torch.Tensor) else np.sin(x)


def cos(x: Value) -> Value:
    return torch.cos(x) if isinstance(x, torch.Tensor) else np.cos(x)


def tan(x: Value) -> Value:
    return torch.tan(x) if isinstance(x, torch.Tensor) else np.tan(x)


def asin(x: Value) -> Value:
    return torch.asin(x) if isinstance(x, torch.Tensor) else np.arcsin(x)


def acos(x: Value) -> Value:
    return torch.acos(x) if isinstance(x, torch.Tensor) else np.arccos(x)


def atan(x: Value) -> Value:
    return torch.atan(x) if isinstance(x, torch.Tensor) else np.arctan(x)


def get_angle(x: Value, y: Value) -> Value:
    angle = -atan(x / y) - (y < 0) * pi + 0.5 * pi
    return angle


def euler_to_matrix(euler_x: float, euler_y: float, euler_z: float) -> list[float]:
    q = glm.quat(glm.radians(glm.vec3(euler_x, euler_y, euler_z)))
    vec_list = glm.transpose(glm.mat3_cast(q)).to_list()
    return vec_list[0] + vec_list[1] + vec_list[2]


def linear_interp(t: Value, range: Value) -> Value:
    return t * (range[1] - range[0]) + range[0]


def norm(value: Tensor, dim: int = -1, keepdim: bool = False) -> Tensor:
    return torch.norm(value, dim=dim, keepdim=keepdim) if isinstance(value, torch.Tensor) \
        else np.linalg.norm(value, axis=dim, keepdims=keepdim)


def normalize(value: Tensor) -> Tensor:
    return value / norm(value, keepdim=True)


def clamp(x: Value, min: Value | float, max: Value | float) -> Value:
    return torch.clamp(x, min, max) if isinstance(x, torch.Tensor) else np.clip(x, min, max)


def smooth_step(x: Value, x0: Value | float, x1: Value | float) -> Value:
    y = clamp((x - x0) / (x1 - x0), 0., 1.)
    return y * y * (3. - 2. * y)


def fov2length(angle: float) -> float:
    """
    Calculate corresponding physical length (at one unit distance) of specified field-of-view (in degrees).

    :param angle `float`: field-of-view in degrees
    :return `float`: physical length at one unit distance
    """
    return tan(radians(angle) / 2) * 2


def length2fov(length: float) -> float:
    """
    Calculate corresponding field-of-view (in degrees) of specified physical length (at one unit distance).

    :param length `float`: physical length at one unit distance
    :return `float`: field-of-view in degrees
    """
    return degrees(atan(length / 2) * 2)
