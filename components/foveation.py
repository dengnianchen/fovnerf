import torch
import torch.nn.functional as F

from utils import img, math
from utils.torch_ext import grid2d
from utils.types import *


class Foveation(object):

    def __init__(self, layers_fov: list[float], layers_res: list[Resolution],
                 out_res: Resolution, *, blend: float = 0.6, device: torch.device = None):
        self.layers_fov = layers_fov
        self.layers_res = layers_res
        self.out_res = out_res
        self.blend = blend
        self.device = device
        self.n_layers = len(self.layers_fov)
        self.eye_fovea_blend = [
            self._gen_layer_blendmap(i)
            for i in range(self.n_layers - 1)
        ]  # blend maps of fovea layers
        self.coords = grid2d(*out_res, device=device)

    def to(self, device: torch.device):
        self.eye_fovea_blend = [x.to(device) for x in self.eye_fovea_blend]
        self.coords = self.coords.to(device)
        return self

    def synthesis(self, layers: list[Tensor], gaze: Tensor, shifts: list[int] = None,
                  do_blend: bool = True) -> Tensor:
        """
        Generate foveated retinal image by blending fovea layers
        **Note: current implementation only support two fovea layers**

        :param layers `List(Tensor(B, C, H'{l}, W'{l}))`: list of foveated layers
        :return `Tensor(B, C, H:out, W:out)`: foveated images
        """
        output = F.interpolate(layers[-1], self.out_res, mode='bilinear', align_corners=False)
        if shifts is not None:
            output = img.shift(output, shifts[-1])
        c = gaze.new_tensor([self.out_res.w, self.out_res.h]) * .5 + gaze
        for i in range(self.n_layers - 2, -1, -1):
            if layers[i] is None:
                continue
            R = self.get_layer_size_in_final_image(i) / 2
            grid = ((self.coords - c) / R)[None, ...]
            if shifts is not None:
                grid = img.shift(grid, shifts[i], -2)
            # (1, 1, H:out, W:out)
            if do_blend:
                blend = F.grid_sample(self.eye_fovea_blend[i][None, None], grid,
                                         align_corners=False)
            else:
                blend = F.grid_sample(torch.ones_like(self.eye_fovea_blend[i][None, None]), grid,
                                         align_corners=False)
            output.mul_(1 - blend).add_(blend * F.grid_sample(layers[i], grid, align_corners=False))
        return output

    def get_layer_size_in_final_image(self, i: int) -> int:
        """
        Get size of layer i in final image

        :param i: index of layer
        :return: size of layer i in final image (in pixels)
        """
        return self.get_source_layer_cover_size_in_target_layer(
            self.layers_fov[i], self.layers_fov[-1], self.out_res[0])

    def get_source_layer_cover_size_in_target_layer(self, source_fov: float, target_fov: float,
                                                    target_pixel_height: int) -> int:
        """
        Get size of layer i in final image

        :param i: index of layer
        :return: size of layer i in final image (in pixels)
        """
        source_physical_height = math.fov2length(source_fov)
        target_physical_height = math.fov2length(target_fov)
        return int(math.ceil(target_pixel_height * source_physical_height / target_physical_height))

    def _gen_layer_blendmap(self, i: int) -> Tensor:
        """
        Generate blend map for fovea layer i

        :param i: index of fovea layer
        :return `Tensor(H{i}, W{i})`: blend map
        """
        size = self.get_layer_size_in_final_image(i)
        R = size / 2
        p = grid2d(size, device=self.device)  # (size, size, 2)
        r = torch.norm(p - R, dim=2)  # (size, size, 2)
        return math.smooth_step(r, R, R * self.blend)

    def get_layers_mask(self, gaze: Tensor = None) -> list[Tensor]:
        """
        Generate mask images for layers
        the meaning of values in mask images:
        -1: skipped
        0~1: blend with inner layer
        1~2: only self layer
        2~3: blend with outer layer

        :return: Mask images for layers
        """
        layers_mask = []
        for i in range(self.n_layers):
            if i == self.n_layers - 1:
                if gaze is None:
                    layers_mask.append(torch.ones(*self.layers_res[i], device=self.device))
                    continue
                c = (gaze.new_tensor([self.out_res.w, self.out_res.h]) * .5 + gaze) / self.out_res.h
            else:
                c = torch.tensor([.5, .5], device=self.device)
            layers_mask.append(torch.full(self.layers_res[i], -1., device=self.device))
            coord = grid2d(*self.layers_res[i], device=self.device) / self.layers_res[i].h
            r = 2 * torch.norm(coord - c, dim=-1)
            inner_radius = self.get_source_layer_cover_size_in_target_layer(
                self.layers_fov[i - 1], self.layers_fov[i], self.layers_res[i].h) / self.layers_res[i].h \
                if i > 0 else 0
            if i == self.n_layers - 1:
                bounds = [inner_radius * (1 - self.blend), inner_radius, 100, 100]
            else:
                bounds = [inner_radius * (1 - self.blend), inner_radius, self.blend, 1]
            for bi in range(len(bounds) - 1):
                region = torch.logical_and(r >= bounds[bi], r < bounds[bi + 1])
                layers_mask[i][region] = bi + \
                    (r[region] - bounds[bi]) / (bounds[bi + 1] - bounds[bi])
        return layers_mask
