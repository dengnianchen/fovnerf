import math
import torch


def gaussian(window_size: int, sigma: float, dtype: torch.dtype) -> torch.Tensor:
    gauss = torch.tensor([
        math.exp(-(x - window_size // 2)**2 / float(2. * sigma**2.))
        for x in range(window_size)
    ], dtype=dtype)
    return gauss / gauss.sum()


def create_window(window_size: int, channel: int, dtype: torch.dtype) -> torch.Tensor:
    _1D_window = gaussian(window_size, 1.5, dtype).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).unsqueeze(0).unsqueeze(0)
    return _2D_window.expand(channel, 1, window_size, window_size).contiguous()


def _ssim(img1: torch.Tensor, img2: torch.Tensor, window: torch.Tensor, window_size: int,
          channel: int, size_average: bool = True) -> torch.Tensor:
    mu1 = torch.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = torch.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = torch.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = torch.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = torch.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01**2
    C2 = 0.03**2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
        ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    return ssim_map.mean() if size_average else ssim_map.mean(1).mean(1).mean(1)


class SSIM(torch.nn.Module):
    window_size: int
    size_average: bool
    channel: int
    window: torch.Tensor

    def __init__(self, window_size: int = 11, size_average: bool = True):
        super().__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = None
        self.window = None

    def forward(self, img1: torch.Tensor, img2: torch.Tensor) -> torch.Tensor:
        if self.window == None or self.channel != img1.shape[1] or self.window.dtype != img1.dtype:
            self.channel = img1.shape[1]
            self.window = create_window(self.window_size, self.channel, img1.dtype)
        self.window = self.window.to(img1.device)
        return _ssim(img1, img2, self.window, self.window_size, self.channel, self.size_average)


default_metric_obj: SSIM = None


def ssim(img1: torch.Tensor, img2: torch.Tensor, window_size: int = 11, size_average: bool = True) -> torch.Tensor:
    global default_metric_obj
    if default_metric_obj is None or default_metric_obj.window_size != window_size\
            or default_metric_obj.size_average != size_average:
        default_metric_obj = SSIM(window_size, size_average)
    return default_metric_obj(img1, img2)
