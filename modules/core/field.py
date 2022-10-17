from ..__common__ import *

__all__ = ["Field"]


class Field(nn.Module):
    def __init__(self, x_chns: int, shape: list[int], skips: list[int] = [], act: str = 'relu'):
        super().__init__({"x": x_chns}, {"f": shape[1]})
        self.net = nn.FcBlock(x_chns, 0, *shape, skips, act)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


