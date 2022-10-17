from ..__common__ import *

__all__ = ["ColorDecoder", "BasicColorDecoder", "NeRFColorDecoder"]


class ColorDecoder(nn.Module):
    def __init__(self, f_chns: int, d_chns: int, color_chns: int):
        super().__init__({"f": f_chns, "d": d_chns}, {"color": color_chns})

    def __call__(self, f: torch.Tensor, d: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError()

    @staticmethod
    def create(f_chns: int, d_chns: int, color_chns: int, type: str, args: dict[str, Any]) -> "ColorDecoder":
        return getattr(sys.modules[__name__], f"{type}ColorDecoder")(
            f_chns=f_chns, d_chns=d_chns, color_chns=color_chns, **args)


class BasicColorDecoder(ColorDecoder):
    def __init__(self, f_chns: int, color_chns: int, out_act: str = "sigmoid", **kwargs):
        super().__init__(f_chns, 0, color_chns)
        self.net = nn.FcLayer(f_chns, color_chns, out_act)

    def __call__(self, f: torch.Tensor, d: torch.Tensor) -> torch.Tensor:
        return self.net(f)


class NeRFColorDecoder(ColorDecoder):
    def __init__(self, f_chns: int, d_chns: int, color_chns: int, act: str = "relu",
                 out_act: str = "sigmoid", **kwargs):
        super().__init__(f_chns, d_chns, color_chns)
        self.feature_layer = nn.FcLayer(f_chns, f_chns)
        self.net = nn.FcBlock(f_chns + d_chns, color_chns, 1, f_chns // 2, [], act, out_act)

    def __call__(self, f: torch.Tensor, d: torch.Tensor) -> torch.Tensor:
        return self.net(union(self.feature_layer(f), d))
