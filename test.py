import argparse
import json
import torch
from concurrent.futures import ThreadPoolExecutor
from matplotlib import pyplot as plt
from collections import defaultdict
from tqdm import trange

from model import Model
from utils import device, img, netio, math
from utils.metrics import ssim, mse, psnr, LPIPS
from utils.types import *
from utils.view import Trans
from data import *
from components.render import render, render_stereo


parser = argparse.ArgumentParser()
parser.add_argument("--media", type=str, default="image",
                    help="Specify the media of output (image|video)")
parser.add_argument("--views", type=lambda s: range(*[int(val) for val in s.split("-")]),
                    help="Specify the range of views to test")
parser.add_argument("--batch", type=int,
                    help="Batch size (to avoid out-of-memory), defaults to pixels of a view")
parser.add_argument("--stereo", type=float, default=0,
                    help="Specify the stereo IPD. If greater than 0, stereo images will be generated")
parser.add_argument("ckpt", type=Path,
                    help="Path to the checkpoint file")
parser.add_argument("dataset", type=Path,
                    help="Path to the dataset description file")
args = parser.parse_args()

# Load model
ckpt = netio.load_checkpoint(args.ckpt)
ckpt_path: Path = ckpt["path"]
model = Model.create(ckpt["args"]["model"], ckpt["args"]["model_args"])
model.load_state_dict(ckpt["states"]["model"])
model.to(device.default()).eval()
print("Model:", str(ckpt_path))
print("Arguments:", json.dumps(ckpt["args"]["model_args"]))

# Load dataset
dataset = Dataset(args.dataset, views_to_load=args.views, device=device.default())
print(f"Dataset: {dataset.root}/{dataset.name}")

run_dir = ckpt_path.parent
out_dir = run_dir / f"output_{ckpt_path.stem.split('_')[-1]}"
out_id = dataset.name
batch_size = args.batch or dataset.pixels_per_view
n = len(dataset)
executor = ThreadPoolExecutor(8)
lpips = LPIPS().to(device.default())
render_types = ["color"]
metrics = args.stereo == 0 and args.media == "image" and defaultdict(list, dummy=[])
video_frames = defaultdict(list) if args.media == "video" else None
out_dir.mkdir(parents=True, exist_ok=True)


def save_image(out: Tensor, out_type: str, view_idx: int):
    out = out.detach().cpu()
    if args.media == 'video':
        video_frames[out_type].append(out)
    else:
        output_subdir = out_dir / f"{out_id}_{out_type}{'_stereo' if args.stereo > 0 else ''}"
        output_subdir.mkdir(parents=True, exist_ok=True)
        executor.submit(img.save, out, output_subdir / f"{view_idx:04d}.png")


def save_error_image(gt: Tensor, out: Tensor, view_idx: int):
    error_image = (mse(out, gt, reduction='none').mean(-3, True) / 1e-2).clamp(0, 1)
    error_image = img.torch2np(error_image)[..., 0]
    output_subdir = out_dir / f"{out_id}_error"
    output_subdir.mkdir(exist_ok=True)

    def save_fn(error_image, view_idx):
        img.save(plt.get_cmap("jet")(error_image), output_subdir / f"{view_idx:04d}.png")
    executor.submit(save_fn, error_image, view_idx)


# Render views in dataset
for i in trange(n, desc="Test"):
    if metrics:
        start_event = torch.cuda.Event(True)
        end_event = torch.cuda.Event(True)
        start_event.record()

    view_idx = dataset.indices[i].item()
    view = Trans(dataset.centers[i], dataset.rots[i])
    if args.stereo > 0:
        out_left, out_right = render_stereo(model, dataset.cam, view, *render_types, ipd=args.stereo,
                                            batch_size=batch_size)
        out = RenderOut({
            key: torch.cat([out_left[key], out_right[key]], dim=2)
            for key in out_left if isinstance(out_left[key], Tensor)
        })
    else:
        out = render(model, dataset.cam, view, *render_types, batch_size=batch_size)

    if metrics:
        end_event.record()
        torch.cuda.synchronize()
        metrics["view"].append(view_idx)
        metrics["time"].append(start_event.elapsed_time(end_event))
        gt_image = dataset.load_images(view_idx)
        out_image = out.color.movedim(-1, -3)
        if gt_image is not None:
            metrics["mse"].append(mse(out_image, gt_image).item())
            metrics["psnr"].append(psnr(out_image, gt_image).item())
            metrics["ssim"].append(ssim(out_image, gt_image).item() * 100)
            metrics["lpips"].append(lpips(out_image, gt_image).item())
            save_error_image(gt_image, out_image, view_idx)
        else:
            metrics["mse"].append(math.nan)
            metrics["psnr"].append(math.nan)
            metrics["ssim"].append(math.nan)
            metrics["lpips"].append(math.nan)

    for key, value in out.items():
        if isinstance(value, Tensor):
            save_image(value, key, view_idx)

if metrics:
    perf_mean_time = sum(metrics['time']) / n
    perf_mean_error = sum(metrics['mse']) / n
    perf_name = f'eval_{out_id}_{perf_mean_time:.1f}ms_{perf_mean_error:.2e}.csv'

    # Remove old performance reports
    for file in out_dir.glob(f'eval_{out_id}*'):
        file.unlink()

    # Save new performance reports
    with (out_dir / perf_name).open('w') as fp:
        fp.write('View, PSNR, SSIM, LPIPS\n')
        fp.writelines([
            f"{metrics['view'][i]}, {metrics['psnr'][i]:.2f}, "
            f"{metrics['ssim'][i]:.2f}, {metrics['lpips'][i]:.2e}\n"
            for i in range(n)
        ])

# Save video if output media is "video"
if args.media == "video":
    for key, frames in video_frames.items():
        output_video_name = f"{out_id}_{key}{'_stereo' if args.stereo > math.tiny else ''}.mp4"
        img.save_video(torch.cat(frames, 0), out_dir / output_video_name, fps=30)
