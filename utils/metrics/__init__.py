from torch.nn import L1Loss as L1, MSELoss as MSE
from torch.nn.functional import l1_loss as l1, mse_loss as mse
from .vgg import vgg, VGG
from .cauchy import cauchy, Cauchy
from .psnr import psnr, PSNR
from .ssim import ssim, SSIM
from .lpips import lpips, LPIPS
