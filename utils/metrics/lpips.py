import torch
import lpips as lp


class LPIPS(torch.nn.Module):

    def __init__(self, net: str = "alex") -> None:
        super().__init__()
        self.fn = lp.LPIPS(net)

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self.fn(input * 2. - 1., target * 2. - 1.)


default_metric_obj = None


def lpips(input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    global default_metric_obj
    if default_metric_obj is None:
        default_metric_obj = LPIPS().to(input.device)
    return default_metric_obj(input, target)
