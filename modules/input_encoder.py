from .__common__ import *

__all__ = ["InputEncoder", "LinearEncoder", "FreqEncoder"]


class InputEncoder(nn.Module):
    """
    Base class for input encoder.
    """

    def __init__(self, in_chns: int, out_chns: int):
        super().__init__({"_": in_chns}, {"_": out_chns})

    def __call__(self, x: Tensor) -> Tensor:
        """
        Encode the input tensor.

        :param x `Tensor(N..., D)`: D-dim inputs
        :return `Tensor(N..., E)`: encoded outputs
        """
        raise NotImplementedError()

    @staticmethod
    def create(chns: int, type: str, args: dict[str, Any]) -> "InputEncoder":
        """
        Create an input encoder of `type` with `args`.

        :param chns `int`: input channels
        :param type `str`: type of input encoder, without suffix "Encoder"
        :param args `{str:Any}`: arguments for initializing the input encoder
        :return `InputEncoder`: the created input encoder
        """
        return getattr(sys.modules[__name__], f"{type}Encoder")(chns, **args)


class LinearEncoder(InputEncoder):
    """
    The linear encoder: D -> D.
    """

    def __init__(self, chns):
        super().__init__(chns, chns)

    def __call__(self, x: Tensor) -> Tensor:
        return x

    def extra_repr(self) -> str:
        return f"{self.in_chns} -> {self.out_chns}"


class FreqEncoder(InputEncoder):
    """
    The frequency encoder introduced in [mildenhall2020nerf]: D -> 2LD[+D].
    """
    
    freq_bands: Tensor
    """
    `Tensor(L)` Frequency bands (1, 2, ..., 2^(L-1))
    """

    def __init__(self, chns, freqs: int, include_input: bool):
        super().__init__(chns, chns * (freqs * 2 + include_input))
        self.include_input = include_input
        self.freqs = freqs
        self.register_buffer("freq_bands", (2. ** torch.arange(freqs))[:, None].expand(-1, chns),
                             persistent=False)

    def __call__(self, x: Tensor) -> Tensor:
        x_ = x.unsqueeze(-2) * self.freq_bands
        result = union(torch.sin(x_), torch.cos(x_)).flatten(-2)
        return union(x, result) if self.include_input else result

    def extra_repr(self) -> str:
        return f"{self.in_chns} -> {self.out_chns}"\
            f"(2x{self.freqs}x{self.in_chns}{f'+{self.in_chns}' * self.include_input})"
