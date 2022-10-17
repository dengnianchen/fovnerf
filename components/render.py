import torch

from model import Model
from utils import math, torch_ext
from utils.view import *
from utils.types import *


def render(model: Model, cam: Camera, view: Trans, *output_types: str,
           gaze_dir: Tensor = None, layer_mask: Tensor = None, batch_size: int = None) -> RenderOut:
    if len(output_types) == 0:
        raise ValueError("'output_types' is empty")

    if gaze_dir is not None:
        local_rays = cam.local_rays + torch_ext.union(gaze_dir, 0.)  # (H*W, 3)
    else:
        local_rays = cam.local_rays
    rays_d = view.trans_vector(local_rays)  # (B..., H*W, 3)
    rays_o = view.t[..., None, :].expand_as(rays_d)
    input = Rays(rays_o=rays_o, rays_d=rays_d)  # (B..., H*W)

    if layer_mask is not None:
        selector = layer_mask.flatten().ge(0).nonzero(as_tuple=True)[0]
        input = input.transform(lambda value: value.index_select(len(input.shape) - 1, selector))
    input = input.flatten()  # (B..., X) -> (N)

    output = RenderOut()  # will be (N)
    n = input.shape[0]
    batch_size = batch_size or n
    for offset in range(0, n, batch_size):
        batch_slice = slice(offset, min(offset + batch_size, n))
        batch_output = model(input.select(batch_slice), *output_types)
        for key, value in batch_output.items():
            match value:
                case Tensor():
                    if key not in output:
                        output[key] = value.new_full([n, *value.shape[1:]],
                                                     math.huge * (key == "depth"))
                    output[key][batch_slice] = batch_output[key]
                case float() | int():
                    output[key] = output.get(key, 0) + value
                case _:
                    output[key] = output.get(key, []) + [value]

    output = output.reshape(*view.shape, -1)  # (N) -> (B..., X)
    if layer_mask is not None:
        output = output.transform(
            lambda value: value.new_zeros(
                *view.shape, local_rays.shape[0], *value.shape[len(view.shape) + 1:])
            .index_copy(len(view.shape), selector, value))
    return output.reshape(*view.shape, *cam.res)  # (B..., H*W) -> (B..., H, W)


def render_stereo(model: Model, cam: Camera, view: Trans, ipd: float, *output_types: str,
                  gaze_dirs: Tensor | tuple[Tensor, Tensor] = (None, None),
                  layer_mask: Tensor = None, batch_size: int = None) -> tuple[RenderOut, RenderOut]:
    left_view = Trans(view.trans_point(view.t.new_tensor([-ipd / 2., 0., 0.])), view.r)
    right_view = Trans(view.trans_point(view.t.new_tensor([ipd / 2., 0., 0.])), view.r)
    return render(model, cam, left_view, *output_types, gaze_dir=gaze_dirs[0], layer_mask=layer_mask,
                  batch_size=batch_size), \
        render(model, cam, right_view, *output_types, gaze_dir=gaze_dirs[1], layer_mask=layer_mask,
               batch_size=batch_size)
