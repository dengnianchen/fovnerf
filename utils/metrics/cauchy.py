import torch


def cauchy(input: torch.Tensor, target: torch.Tensor = None, *, s=1.0, sum=False) -> torch.Tensor:
    x = input - target if target is not None else input
    y = (s * x * x * 0.5 + 1).log()
    return y.sum() if sum else y.mean()


class Cauchy(torch.nn.Module):

    def __init__(self, s=1.0, sum=False):
        super().__init__()
        self.s = s
        self.sum = sum

    def forward(self, input: torch.Tensor, target: torch.Tensor = None) -> torch.Tensor:
        return cauchy(input, target, s=self.s, sum=self.sum)
