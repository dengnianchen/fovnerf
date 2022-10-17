from ..__common__ import *
from .field import *
from .color_decoder import *
from .density_decoder import *


class NeRF(nn.Module):

    def __init__(self, x_chns: int, d_chns: int, color_chns: int, depth: int, width: int,
                 skips: list[int], act: str, color_decoder_type: str):
        """
        Initialize a NeRF core module.

        :param x_chns `int`: channels of input positions (D_x)
        :param d_chns `int`: channels of input directions (D_d)
        :param color_chns `int`: channels of output colors (D_c)
        :param depth `int`: number of layers in field network
        :param width `int`: width of each layer in field network
        :param skips `[int]`: skip connections from input to specific layers in field network
        :param act `str`: activation function in field network and color decoder
        :param color_decoder_type `str`: type of color decoder
        """
        super().__init__({"x": x_chns, "d": d_chns}, {"density": 1, "color": color_chns})
        self.field = Field(x_chns, [depth, width], skips, act)
        self.density_decoder = DensityDecoder(self.field.out_chns, 1)
        self.color_decoder = ColorDecoder.create(self.field.out_chns, d_chns, color_chns,
                                                 color_decoder_type, {"act": act})

    # stub method for type hint
    def __call__(self, x: torch.Tensor, d: torch.Tensor) -> torch.Tensor:
        """
        Inference colors and densities from input samples

        :param x `Tensor(B..., D_x)`: input positions
        :param d `Tensor(B..., D_d)`: input directions
        :return `Tensor(B..., D_c + D_σ)`: output colors and densities
        """
        f = self.field(x)
        densities = self.density_decoder(f)
        colors = self.color_decoder(f, d)
        return torch.cat([colors, densities], -1)
