from ..__common__ import *
from .field import *
from .color_decoder import *
from .density_decoder import *


class FovNeRF(nn.Module):

    def __init__(self, x_chns: int, color_chns: int, depth: int, width: int,
                 skips: list[int], act: str, n_samples: int, n_fields: int):
        """
        Initialize a FS-NeRF core module.

        :param x_chns `int`: channels of input positions (D_x)
        :param d_chns `int`: channels of input directions (D_d)
        :param color_chns `int`: channels of output colors (D_c)
        :param depth `int`: number of layers in field network
        :param width `int`: width of each layer in field network
        :param skips `[int]`: skip connections from input to specific layers in field network
        :param act `str`: activation function in field network and color decoder
        :param color_decoder_type `str`: type of color decoder
        """
        super().__init__({"x": x_chns}, {"rgbd": 1 + color_chns})
        self.n_fields = n_fields
        self.color_chns = color_chns
        self.samples_per_field = n_samples // n_fields
        self.subnets = torch.nn.ModuleList()
        for _ in range(n_fields):
            field = Field(x_chns * self.samples_per_field, [depth, width], skips, act)
            density_decoder = DensityDecoder(field.out_chns, self.samples_per_field)
            color_decoder = BasicColorDecoder(field.out_chns, color_chns * self.samples_per_field)
            self.subnets.append(torch.nn.ModuleDict({
                "field": field,
                "density_decoder": density_decoder,
                "color_decoder": color_decoder
            }))

    def __call__(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Inference colors and densities from input samples

        :param x `Tensor(B, P, D_x)`: input positions
        :return `Tensor(B, P, D_c + D_σ)`: output colors and densities
        """
        densities = []
        colors = []
        for i in range(self.n_fields):
            f = self.subnets[i]["field"](
                x[:, i * self.samples_per_field:(i + 1) * self.samples_per_field].flatten(-2))
            densities.append(self.subnets[i]["density_decoder"](f).reshape(
                    -1, self.samples_per_field, 1))
            colors.append(self.subnets[i]["color_decoder"](f, None).reshape(
                    -1, self.samples_per_field, self.color_chns))
        return union(torch.cat(colors, -2), torch.cat(densities, -2))
