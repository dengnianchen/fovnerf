import torch

from utils.view import *
from utils.types import *
from utils import math
from utils.nn import Module
from .foveation import Foveation
from .render import render, render_stereo


class FoveatedNeuralRenderer(object):

    def __init__(self, layers_fov: list[float], layers_res: list[Resolution],
                 layers_net: list[Module], output_res: Resolution):
        super().__init__()
        self.device = layers_net[0].device
        self.layers_net = layers_net
        self.layers_cam = [
            Camera({"fov": layers_fov[i]}, layers_res[i], device=self.device)
            for i in range(len(layers_fov))
        ]
        self.cam = Camera({"fov": layers_fov[-1]}, output_res, device=self.device)
        self.foveation = Foveation(layers_fov, layers_res, output_res, device=self.device)

    def __call__(self, view: Trans, gazes: Tensor, ipd: float = 0, using_mask: bool = True,
                 stereo_adapt: bool = True) -> dict[str, Tensor] | tuple[dict[str, Tensor], dict[str, Tensor]]:
        layers_mask = self.foveation.get_layers_mask() if using_mask else [None] * 3
        gazes = gazes.expand(2, 2)
        gaze_dirs = gazes / self.cam.f
        mean_gaze = .5 * (gazes[0] + gazes[1])
        mean_gaze_dir = .5 * (gaze_dirs[0] + gaze_dirs[1])

        if ipd > math.tiny:
            periph_shift = int((gazes[0, 0] - gazes[1, 0]).item()) // 2 if stereo_adapt else 0

            if stereo_adapt:
                stereo_fovea = render_stereo(self.layers_net[0], self.layers_cam[0], view, ipd,
                                             "color", gaze_dirs=gaze_dirs, layer_mask=layers_mask[0])
                mid = render(self.layers_net[1], self.layers_cam[1], view, "color",
                             gaze_dir=mean_gaze_dir, layer_mask=layers_mask[1])
                periph = render(self.layers_net[2], self.layers_cam[2], view, "color")
                layers_left = [stereo_fovea[0], mid, periph]
                layers_right = [stereo_fovea[1], mid, periph]
            else:
                stereo_layers = [
                    render_stereo(self.layers_net[i], self.layers_cam[i], view, ipd, "color",
                                  gaze_dirs=gaze_dirs if i < 2 else (None, None),
                                  layer_mask=layers_mask[i])
                    for i in range(3)
                ]
                layers_left = [stereo_layers[i][0] for i in range(3)]
                layers_right = [stereo_layers[i][1] for i in range(3)]

            return self._gen_output(layers_left, gazes[0], [0, 0, periph_shift]),\
                self._gen_output(layers_right, gazes[1], [0, 0, -periph_shift])
        else:
            layers = [
                render(self.layers_net[i], self.layers_cam[i], view, "color",
                       gaze_dir=mean_gaze_dir if i < 2 else None, layer_mask=layers_mask[i])
                for i in range(3)
            ]
            return self._gen_output(layers, mean_gaze)

    def _gen_output(self, layers: list[RenderOut], gaze: tuple[float, float], shifts: list[int] = None) -> dict[str, Tensor]:
        layers_img = self._post_process([layer["color"].movedim(-1, -3) for layer in layers])
        blended = self.foveation.synthesis(layers_img, gaze, shifts)
        return {
            'layers_img': layers_img,
            'blended': blended
        }

    def _post_process(self, layers_img: list[Tensor]) -> list[Tensor]:
        def constrast_enhance(image: Tensor, sigma: float, fe: float) -> Tensor:
            kernel = torch.ones(1, 1, 3, 3, device=image.device) / 9
            mean = torch.cat([
                torch.conv2d(image[:, 0:1], kernel, padding=1),
                torch.conv2d(image[:, 1:2], kernel, padding=1),
                torch.conv2d(image[:, 2:3], kernel, padding=1)
            ], 1)
            cScale = 1.0 + sigma * fe
            return torch.clamp(mean + (image - mean) * cScale, 0, 1)
        return [
            constrast_enhance(layers_img[0], 3, 0.2),
            constrast_enhance(layers_img[1], 5, 0.2),
            constrast_enhance(layers_img[2], 5, 0.2)
        ]
