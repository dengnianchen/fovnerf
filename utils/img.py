import os
import shutil
import torch
import uuid
import matplotlib.pyplot as plt
import numpy as np
from .types import *


def is_image_file(filename):
    """
    Chech if `filename` is an image file (with extension of .png, .jpg or .jpeg)

    :param filename `str`: name of the file to check
    :return `bool`: whether `filename` is an image file or not
    """
    return any(filename.endswith(extension) for extension in [".png", ".jpg", ".jpeg"])


def np2torch(img: np.ndarray, permute: bool = True) -> Tensor:
    """
    Convert numpy-images(s) to torch-image(s), permute channels dim if `permute=True`

    :param input `ndarray([B]HWC)`: 3D or 4D numpy-image(s)
    :return `Tensor([B]HWC|[B]CHW)`: 3D or 4D torch-image(s)
    """
    batch_input = len(img.shape) == 4
    if permute:
        t = torch.from_numpy(np.transpose(img, [0, 3, 1, 2] if batch_input else [2, 0, 1]))
    else:
        t = torch.from_numpy(img)
    if not batch_input:
        t = t.unsqueeze(0)
    return t


def torch2np(input: Tensor) -> np.ndarray:
    """
    Convert torch-image(s) to numpy-images(s) with channels at the last dim

    :param input `Tensor(HW|[B]CHW|[B]HWC)`: 2D, 3D or 4D torch-image(s)
    :return `ndarray ([B]HWC)`: numpy-image(s) with channels transposed to the last dim
    """
    img = input.cpu().detach().numpy()
    if len(img.shape) == 2:  # 2D(HW): Single channel image
        return img
    batch_input = len(img.shape) == 4
    if input.size()[batch_input] <= 4:  # 3D(CHW) or 4D(BCHW): transpose channel
        return np.transpose(img, [0, 2, 3, 1] if batch_input else [1, 2, 0])
    return img


def load(*paths: Path | MultiPaths, permute=True, with_alpha=False) -> Tensor:
    """
    Load one or multiple torch-image(s).

    :param paths `Path|MultiPaths...`: path(s) of image(s) to load
    :param permute `bool?`: whether permute channels dim, defaults to `True`
    :param with_alpha `bool?`:whether load the alpha channel, defaults to `False`
    :return `Tensor(BCHW|BHWC)`: loaded torch-image(s)
    """
    paths = sum(([item] if isinstance(item, Path) else list(item) for item in paths), start=[])
    chns = 4 if with_alpha else 3
    imgs = np.stack([plt.imread(path)[..., :chns] for path in paths])
    if imgs.dtype == 'uint8':
        imgs = imgs.astype(np.float32) / 255
    return np2torch(imgs, permute)


def save(input: Tensor | np.ndarray, *paths: Path):
    """
    Save one or multiple image(s).

    If `input` contains multiple images and `paths` has only one argument specified, this method
    will try to treat the argument as a path pattern and save those images in sequence.

    :param input `Tensor|ndarray`: image(s) to save
    :param *paths `str...`: path(s) to save image(s) to
    :raises `ValueError`: if number of paths does not match number of input image(s)
    """
    paths = sum(([item] if isinstance(item, Path) else list(item) for item in paths), start=[])
    if len(input.shape) < 4:
        input = input[None]
    if input.shape[0] != len(paths):
        if len(paths) == 1:
            paths = [*(paths[0] % range(input.shape[0]))]
        else:
            raise ValueError("Number of path(s) does not match number of input image(s)")
    np_img = torch2np(input) if isinstance(input, Tensor) else input
    if np_img.dtype.kind == 'f':
        np_img = np.clip(np_img, 0, 1)
    if np_img.shape[-1] == 1:
        np_img = np.repeat(np_img, 3, axis=-1)
    if not np_img.flags['C_CONTIGUOUS']:
        np_img = np.ascontiguousarray(np_img)
    for i, path in enumerate(paths):
        plt.imsave(path, np_img[i])


def save_video(frames: Tensor, path: Path, fps: int, repeat: int = 1, pingpong: bool = False):
    """
    Encode and save a sequence of frames to a video file.

    :param frames `Tensor(B, C, H, W)`: a sequence of frames
    :param path `Path`: video path
    :param fps `int`: frames per second
    :param repeat `int?`: repeat times, defaults to `1`
    :param pingpong `bool?`: whether repeat sequence in pinpong form, defaults to `False`
    """
    if pingpong:
        frames = torch.cat([frames, frames.flip(0)], 0)
    if repeat > 1:
        frames = frames.expand(repeat, -1, -1, -1, -1).flatten(0, 1)

    path = Path(path)
    tempdir = Path(f'/dev/shm/dvs_tmp/video/{uuid.uuid4().hex}')
    temp_frame_file_pattern = tempdir / f"%04d.bmp"
    path.parent.mkdir(parents=True, exist_ok=True)
    tempdir.mkdir(parents=True, exist_ok=True)

    save(frames, temp_frame_file_pattern)
    os.system(f'ffmpeg -y -r {fps:d} -i {temp_frame_file_pattern} -c:v libx264 {path}')
    shutil.rmtree(tempdir)


def plot(input: Tensor | np.ndarray, *, ax: plt.Axes = None):
    """
    Plot a torch-image using matplotlib

    :param input `Tensor(HW|[B]CHW|[B]HWC)`: 2D, 3D or 4D torch-image(s)
    :param ax `plt.Axes`: (optional) specify the axes to plot image
    """
    im = torch2np(input) if isinstance(input, Tensor) else input
    if len(im.shape) == 4:
        im = im[0]
    return plt.imshow(im) if ax is None else ax.imshow(im)


def shift(input: Tensor, offset: int, dim=-1) -> Tensor:
    if offset == 0:
        return input
    shifted = torch.zeros_like(input)
    if dim < 0:
        dim = len(input.shape) + dim
    src_index = [slice(None)] * len(input.shape)
    tgt_index = [slice(None)] * len(input.shape)
    if offset > 0:
        src_index[dim] = slice(-offset)
        tgt_index[dim] = slice(offset, None)
    else:
        src_index[dim] = slice(-offset, None)
        tgt_index[dim] = slice(offset)
    shifted[tgt_index] = input[src_index]
    return shifted
