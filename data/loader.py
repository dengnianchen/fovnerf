import torch
import torch.utils.data
from tqdm import tqdm
from collections import defaultdict

from .dataset import Dataset
from utils import math
from utils.types import *


class RaysLoader(object):

    class Iterator(object):

        def __init__(self, loader: "RaysLoader"):
            super().__init__()
            self.loader = loader
            self.offset = 0

            # Initialize ray indices
            self.ray_indices = torch.randperm(self.loader.tot_pixels, device="cpu")\
                if loader.shuffle else torch.arange(self.loader.tot_pixels, device="cpu")

        def __next__(self) -> Rays:
            if self.offset >= self.ray_indices.shape[0]:
                raise StopIteration()
            stop = min(self.offset + self.loader.batch_size, self.ray_indices.shape[0])
            rays = self._get_rays(self.ray_indices[self.offset:stop])
            self.offset = stop
            return rays

        def _get_rays(self, indices: Tensor) -> Rays:
            indices_on_device = indices.to(self.loader.device) # (B)
            view_idx = torch.div(indices_on_device, self.loader.pixels_per_view, rounding_mode="trunc")
            pix_idx = indices_on_device % self.loader.pixels_per_view
            rays_o = self.loader.centers[view_idx]  # (B, 3)
            rays_d = self.loader.local_rays[pix_idx]  # (B, 3)
            if self.loader.rots is not None:
                rays_d = (self.loader.rots[view_idx] @ rays_d[..., None])[..., 0]
            rays = Rays({
                'idx': indices_on_device,
                'rays_o': rays_o,
                'rays_d': rays_d
            })

            # "colors" and "depths" are on host memory. Move part of them to device memory
            indices = indices.to("cpu")
            if "color" in self.loader.data:
                rays["color"] = self.loader.data["color"][indices].to(self.loader.device, non_blocking=True)
            return rays

    def __init__(self, dataset: Dataset, batch_size: int, *,
                 shuffle: bool = False, num_workers: int = 4, device: torch.device = None):
        super().__init__()
        self.dataset = dataset
        self.batch_size = batch_size
        self.device = device
        self.shuffle = shuffle

        self.n_views = len(dataset)
        self.pixels_per_view = dataset.pixels_per_view
        self.tot_pixels = self.n_views * self.pixels_per_view

        self.indices = dataset.indices.to(self.device)
        self.centers = dataset.centers.to(self.device)
        self.rots = dataset.rots.to(self.device) if dataset.rots is not None else None
        self.local_rays = dataset.cam.local_rays.to(self.device)

        # Load views from dataset
        self.data = defaultdict(list)
        views_loader = torch.utils.data.DataLoader(dataset, num_workers=num_workers,
                                                   pin_memory=True)
        for view_data in tqdm(views_loader, "Loading views", leave=False, dynamic_ncols=True):
            for key, val in view_data.items():
                self.data[key].append(val)
        print(f"{len(dataset)} views loaded.")
        self.data = {
            key: torch.cat(val).flatten(0, 1)
            for key, val in self.data.items() if key == "color"
        }

    def __iter__(self):
        return RaysLoader.Iterator(self)

    def __len__(self):
        return math.ceil(self.tot_pixels / self.batch_size)
