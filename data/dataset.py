import json
import torch
import torch.utils.data
import torch.nn.functional as nn_f
from typing_extensions import Self

from utils.types import *
from utils.view import Camera
from utils import img, math
from utils.misc import calculate_autosize


class DataDesc(dict[str, Any]):
    path: Path

    @property
    def name(self) -> str:
        return self.path.stem

    @property
    def root(self) -> Path:
        return self.path.parent

    @property
    def coord_sys(self) -> str:
        return "gl" if self.get("gl_coord") else "dx"

    def __init__(self, path: Path):
        path = DataDesc.get_json_path(path)
        with open(path, 'r', encoding='utf-8') as file:
            data = json.loads(file.read())
        super().__init__(data)
        self.path = path

    @staticmethod
    def get_json_path(path: Path) -> Path:
        if path.suffix != ".json":
            path = Path(f"{path}.json")
        return path.absolute()

    def get(self, key: str, fn=lambda x: x, default=None) -> Any | None:
        if key in self:
            return fn(self[key])
        return default

    def get_as_tensor(self, key: str, fn=lambda x: x, default=None, dtype=torch.float,
                      device: torch.device = None, shape: torch.Size = None) -> torch.Tensor | None:
        raw_value = self.get(key, fn, default)
        if raw_value is None:
            return raw_value
        tensor_value = torch.tensor(raw_value, dtype=dtype, device=device)
        if shape is not None:
            tensor_value = tensor_value.reshape(shape)
        return tensor_value

    def get_path(self, name: str) -> Path | None:
        path_pattern = self.get(f"{name}_file")
        if not path_pattern:
            return None
        if "/" not in path_pattern:
            path_pattern = f"{self.name}/{path_pattern}"
        return self.root / path_pattern


class Dataset(torch.utils.data.Dataset):

    root: Path
    """`Path` Root directory of the dataset"""
    name: str
    """`str` Name of the dataset"""
    cam: Camera | None
    """`Camera?` Camera object"""
    device: torch.device
    """`device` Device of tensors"""
    depth_range: tuple[float, float] | None
    """`(float, float)?` Depth range of the scene as a guide to sampling"""
    color_path: Path | None
    """`str?` Path of image data"""
    indices: torch.Tensor
    """`Tensor(N)` Indices for loading specific subset of views in the dataset"""
    centers: torch.Tensor
    """`Tensor(N, 3)` Center positions of views"""
    rots: torch.Tensor
    """`Tensor(N, 3, 3)` Rotation matrices of views"""
    gazes: torch.Tensor | None
    """`Tensor(N, 2, 2)?` Stereo gaze positions (only used for foveated rendering)"""

    @property
    def disparity_range(self) -> tuple[float, float] | None:
        return self.depth_range and (1 / self.depth_range[0], 1 / self.depth_range[1])

    @property
    def pixels_per_view(self) -> int:
        return self.cam.local_rays.shape[0] if self.cam else 0

    @property
    def tot_pixels(self) -> int:
        return len(self) * self.pixels_per_view

    @overload
    def __init__(self, desc: DataDesc | Path, *,
                 res: Resolution = None,
                 views_to_load: IndexSelector = None,
                 device: torch.device = None):
        ...

    @overload
    def __init__(self, dataset: Self, *,
                 views_to_load: IndexSelector = None):
        ...

    def __init__(self, dataset_or_desc: Self | DataDesc | Path, *,
                 res: Resolution = None,
                 views_to_load: IndexSelector = None,
                 device: torch.device = None):
        super().__init__()
        if isinstance(dataset_or_desc, Dataset):
            self._init_from_dataset(dataset_or_desc, views_to_load=views_to_load)
        else:
            self._init_from_desc(dataset_or_desc, res=res, views_to_load=views_to_load,
                                 device=device)

    def __getitem__(self, index: int | torch.Tensor | slice) -> dict[str, torch.Tensor]:
        if isinstance(index, torch.Tensor) and len(index.shape) == 0:  # scalar tensor
            index = index.item()
        view_index = self.indices[index]
        data = {
            "t": self.centers[index],
            "r": self.rots[index]
        }
        image = self.load_images(view_index)
        if image is not None:
            data["color"] = self.cam.get_pixels(image)
            if isinstance(index, int):
                data["color"].squeeze_(0)
        return data

    def __len__(self):
        return self.indices.shape[0]

    def load_images(self, indices: int | list[int] | torch.Tensor) -> torch.Tensor:
        if not self.color_path:
            return None
        if isinstance(indices, torch.Tensor) and len(indices.shape) == 0:
            indices = indices.item()
        images = img.load(self.color_path % indices).to(device=self.device)
        if self.cam.res != list(images.shape[-2:]):
            images = nn_f.interpolate(images, self.cam.res)
        return images

    def split(self, *views: int) -> list[Self]:
        views, _ = calculate_autosize(len(self), *views)
        sub_datasets = []
        offset = 0
        for i in range(len(views)):
            end = offset + views[i]
            sub_datasets.append(Dataset(self, views_to_load=slice(offset, end)))
            offset = end
        return sub_datasets

    def _init_from_desc(self, desc_or_path: DataDesc | Path, res: Resolution = None,
                        views_to_load: IndexSelector = None, device: torch.device = None):
        desc = desc_or_path if isinstance(desc_or_path, DataDesc) else DataDesc(desc_or_path)
        self.root = desc.root
        self.name = desc.name
        self.color_path = desc.get_path("color")
        self.device = device

        self.cam = Camera(desc["cam"], res or Resolution.from_str(desc["res"]), device=device) \
            if "cam" in desc else None
        self.depth_range = desc.get("depth_range")
        self.centers = desc.get_as_tensor("centers", device=device)
        self.rots = desc.get_as_tensor("rots", lambda rots: [
            math.euler_to_matrix(rot[1] if desc.coord_sys == "gl" else -rot[1], rot[0], 0)
            for rot in rots
        ] if len(rots[0]) == 2 else rots, shape=(-1, 3, 3), device=device)
        self.gazes = desc.get_as_tensor("gazes", device=device, shape=(-1, 2, 2))
        self.indices = desc.get_as_tensor("views", default=list(range(self.centers.shape[0])),
                                          dtype=torch.long, device=device)

        if views_to_load is not None:
            if isinstance(views_to_load, list):
                views_to_load = torch.tensor(views_to_load, device=device)
            self.indices = self.indices[views_to_load]
            self.centers = self.centers[views_to_load]
            self.rots = self.rots[views_to_load]
            self.gazes = self.gazes[views_to_load] if self.gazes is not None else None

        if desc.coord_sys != "gl":
            self.centers[:, 2] *= -1
            self.rots[:, 2] *= -1
            self.rots[..., 2] *= -1

    def _init_from_dataset(self, dataset: Self, views_to_load: IndexSelector = None):
        """
        Clone or get subset of an existed dataset

        :param dataset `Dataset`: _description_
        :param views_to_load `IndexSelector?`: _description_, defaults to None
        """
        self.root = dataset.root
        self.name = dataset.name
        self.device = dataset.device
        self.cam = dataset.cam
        self.depth_range = dataset.depth_range
        self.color_path = dataset.color_path
        if views_to_load is not None:
            if isinstance(views_to_load, list):
                views_to_load = torch.tensor(views_to_load, device=dataset.device)
            self.indices = dataset.indices[views_to_load].clone()
            self.centers = dataset.centers[views_to_load].clone()
            self.rots = dataset.rots[views_to_load].clone()
            self.gazes = dataset.gazes[views_to_load].clone() if dataset.gazes is not None else None
        else:
            self.indices = dataset.indices.clone()
            self.centers = dataset.centers.clone()
            self.rots = dataset.rots.clone()
            self.gazes = dataset.gazes.clone() if dataset.gazes is not None else None
