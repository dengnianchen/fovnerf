from .__common__ import *
from .data_pack import *


class Rays(DataPack):
    rays_o: Tensor1D
    """`Tensor(B..., 3)`"""
    rays_d: Tensor1D
    """`Tensor(B..., 3)`"""
    idx: Tensor0D | None
    """`Tensor(B...)`"""
    color: Tensor1D | None
    """`Tensor(B..., C)`"""

    def get_points(self, z: Tensor) -> Tensor:
        """
        Get points along rays at distance z

        :param z `Tensor(B..., P)`: distances along rays
        :return `Tensor(B..., P, 3)`: points along rays
        """
        return self.rays_o[..., None, :] + self.rays_d[..., None, :] * z[..., None]
