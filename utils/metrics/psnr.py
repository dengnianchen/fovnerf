import torch
import math
from torch.nn.functional import mse_loss


def psnr(input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    mse = mse_loss(input, target)
    return -10. * torch.log(mse) / math.log(10.)


class PSNR(torch.nn.Module):

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return psnr(input, target)
