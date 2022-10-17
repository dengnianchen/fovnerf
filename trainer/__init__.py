import sys

from .trainer import Trainer, trainer_classes
from .basic import BasicTrainer

__all__ = ["Trainer", "BasicTrainer"]


# Register all trainer classes
for item in __all__:
    if item != "Trainer":
        trainer_classes[item] = getattr(sys.modules[__name__], item)
