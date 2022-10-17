import os
import argparse
import shutil
import torch
from tqdm import trange
from data.dataset import Dataset

from utils import img, device, math
from utils.types import *
from utils.view import Trans
from components.fnr import FoveatedNeuralRenderer
from model import Model


torch.set_grad_enabled(False)

parser = argparse.ArgumentParser()
parser.add_argument('-i', '--ipd', type=float, default=0.06,
                    help="The stereo IPD. Render mono images/video if this value is zero")
parser.add_argument('-r', '--range', type=str,
                    help="The range of frames to render, specified as format: start[:end[:step]]")
parser.add_argument('-f', '--fps', type=int,
                    help="The FPS of output video. If not specified, a sequence of images will be saved instead")
parser.add_argument('-m', '--model', type=str, required=True,
                    help="The directory containing fovea* and periph* model file")
parser.add_argument('view_file', type=str,
                    help="The path to .csv or .json file which contains a sequence of poses and gazes")
opt = parser.parse_args()


def load_csv(file: Path) -> tuple[Trans, torch.Tensor]:
    with open(file, 'r', encoding='utf-8') as file:
        data = torch.tensor([
            [float(str) for str in line_content.split(',')]
            for line_content in file.readlines()
        ], device=device.default())  # (N, 20)
    gazes = data[:, :4].reshape(-1, 2, 2)
    t = data[:, 4:7]
    r = data[:, 7:].reshape(-1, 3, 3)
    return Trans(t, r), gazes


def load_json(file: Path) -> tuple[Trans, torch.Tensor]:
    dataset = Dataset(file, device=device.default())
    return Trans(dataset.centers, dataset.rots), dataset.gazes


def load_views_and_gazes(data_desc_file: Path) -> tuple[Trans, torch.Tensor]:
    if data_desc_file.suffix == '.csv':
        views, gazes = load_csv(data_desc_file)
    else:
        views, gazes = load_json(data_desc_file)
    gazes[:, :, 1] = (gazes[:, :1, 1] + gazes[:, 1:, 1]) * .5
    return views, gazes


def add_hint(image: Tensor, hint: Tensor, gazes: Tensor):
    if stereo_ipd > math.tiny and len(gazes.shape) == 2:
        add_hint(image[..., :image.size(-1) // 2], hint, gazes[0])
        add_hint(image[..., image.size(-1) // 2:], hint, gazes[1])
        return
    gaze = (.5 * (gazes[0] + gazes[1]) if len(gazes.shape) == 2 else gazes).tolist()
    fovea_origin = (
        int(gaze[0]) + image.size(-1) // 2 - hint.size(-1) // 2,
        int(gaze[1]) + image.size(-2) // 2 - hint.size(-2) // 2
    )
    fovea_region = (
        ...,
        slice(fovea_origin[1], fovea_origin[1] + hint.size(-2)),
        slice(fovea_origin[0], fovea_origin[0] + hint.size(-1)),
    )
    try:
        image[fovea_region] = image[fovea_region] * (1 - hint[:, 3:]) + hint[:, :3] * hint[:, 3:]
    except Exception:
        print(fovea_region, image.shape, hint.shape)
        exit()


stereo_ipd: float = opt.ipd
res_full = Resolution(1600, 1440)
fov_list = [20.0, 45.0, 110.0]
res_list = [Resolution(256, 256), Resolution(256, 256), Resolution(256, 230)]
hint = img.load(Path("fovea_hint.png"), with_alpha=True).to(device=device.default())

# Initialize foveated neural renderer
model_dir = Path(opt.model)
fovea_net = Model.load(next(model_dir.glob("fovea*.tar")), device.default(), True)
periph_net = Model.load(next(model_dir.glob("periph*.tar")), device.default(), True)
renderer = FoveatedNeuralRenderer(fov_list, res_list, [fovea_net] + [periph_net] * 2, res_full)

# Load Dataset
view_file = Path(opt.view_file)
views, gazes = load_views_and_gazes(view_file)
if opt.range:
    view_range = slice(*[None if not val else int(val) for val in opt.range.split(":")])
    views, gazes = views[view_range], gazes[view_range]
n_views = views.shape[0]
print('Dataset loaded. Views:', n_views)

# Setup directories
videodir = view_file.absolute().parent
tempdir = Path('/dev/shm/dvs_tmp/video')
videoname = f"{view_file.stem}_{('stereo' if stereo_ipd > math.tiny else 'mono')}"
if opt.fps:
    inferout = tempdir / videoname / "%04d.bmp"
    hintout = tempdir / f"{videoname}_hint" / "%04d.bmp"
else:
    inferout = videodir / videoname / "%04d.png"
    hintout = videodir / f"{videoname}_hint" / "%04d.png"
os.makedirs(os.path.dirname(inferout), exist_ok=True)
os.makedirs(os.path.dirname(hintout), exist_ok=True)
print("Video dir:", videodir)
print("Infer out:", inferout)
print("Hint out:", hintout)

# Do rendering
for view_idx in trange(n_views):
    view_gazes = gazes[view_idx]
    view_trans = views[view_idx]
    render_out = renderer(view_trans, view_gazes, stereo_ipd)
    if stereo_ipd > math.tiny:
        frame = torch.cat([render_out[0]['blended'], render_out[1]['blended']], -1)
    else:
        frame = render_out['blended']
    img.save(frame, inferout % view_idx)
    add_hint(frame, hint, view_gazes)
    img.save(frame, hintout % view_idx)

if opt.fps:
    # Generate video without hint
    os.system(f'ffmpeg -y -r {opt.fps:d} -i {inferout} -c:v libx264 {videodir}/{videoname}.mp4')

    # Generate video with hint
    os.system(f'ffmpeg -y -r {opt.fps:d} -i {hintout} -c:v libx264 {videodir}/{videoname}_hint.mp4')

    # Clean temp images
    shutil.rmtree(os.path.dirname(inferout))
    shutil.rmtree(os.path.dirname(hintout))
