from ..__common__ import *

__all__ = ["DensityDecoder"]


class DensityDecoder(nn.Module):
    def __init__(self, f_chns: int, density_chns: int):
        super().__init__({"f": f_chns}, {"density": density_chns})
        self.net = nn.FcLayer(f_chns, density_chns)

    def __call__(self, f: torch.Tensor) -> torch.Tensor:
        return self.net(f)
