import sys

from .model import Model, model_classes
from .nerf import NeRF
from .fov_nerf import FovNeRF

__all__ = ["Model", "NeRF", "FovNeRF"]


# Register all model classes
for item in __all__:
    if item != "Model":
        model_classes[item] = getattr(sys.modules[__name__], item)
