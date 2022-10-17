import torch
from inspect import isclass
from operator import countOf
from types import UnionType
from .__common__ import *


class _Tensor(Tensor):
    data_dim: int


class Tensor0D(_Tensor):
    data_dim: int = 0


class Tensor1D(_Tensor):
    data_dim: int = 1


class Tensor2D(_Tensor):
    data_dim: int = 2


class DataPack(dict[str, Tensor | Any]):
    shape: list[int]
    """Shape of the data pack"""

    device: torch.device
    """Device where tensors in the data pack locate"""

    def __init__(self, map: dict[str, Tensor | Any] = None, shape: list[int] = None,
                 **_data: Tensor | Any) -> None:
        super().__init__((map or {}) | _data)
        self.update_shape(shape)

        tensors = {key: val for key, val in self.items() if isinstance(val, Tensor)}
        tensor0 = next(tensors.values().__iter__(), None)
        device = None if tensor0 is None else tensor0.device
        self._validate_device(device, tensors)
        super().__setattr__("device", device)

    def update_shape(self, new_shape: list[int] = None):
        tensors = {key: val for key, val in self.items() if isinstance(val, Tensor)}
        if new_shape is None:
            # Infer shape from tensors in the data pack and type annotations
            first_key = next((key for key in tensors if self._get_data_dim(key) >= 0), None)
            if first_key is None:
                self._infer_shape_from_tensors_only(tensors)
                return
            # Infer batch shape from the first tensor with specified data_dim
            new_shape = self._get_batch_shape(tensors[first_key], self._get_data_dim(first_key))
        self._validate_shape(new_shape, tensors)
        super().__setattr__("shape", new_shape)

    def transform(self, fn: Callable[[Tensor], Tensor | Any]):
        return self.__class__(**{
            key: fn(val) if isinstance(val, Tensor) else val
            for key, val in self.items()
        })

    def select(self, index: IndexSelector):
        if isinstance(index, Tensor) and index.dtype == torch.bool:
            index = index.nonzero(as_tuple=True)
        return self.transform(lambda tensor: tensor[index])

    def reshape(self, *shape: int):
        return self.transform(lambda tensor: tensor.reshape(*shape, *tensor.shape[len(self.shape):]))

    def flatten(self, start_dim: int = 0, end_dim: int = -1):
        if start_dim < 0:
            start_dim += len(self.shape)
        if end_dim < 0:
            end_dim += len(self.shape)
        if start_dim < 0 or start_dim >= len(self.shape) - 1:
            raise ValueError("\"start_dim\" is out of range")
        if end_dim < 1 or end_dim >= len(self.shape):
            raise ValueError("\"end_dim\" is out of range")
        return self.transform(lambda tensor: tensor.flatten(start_dim, end_dim))

    def to(self, device: torch.device):
        return self.transform(lambda tensor: tensor.to(device))

    def __getattr__(self, __name: str) -> Tensor | Any:
        try:
            return self[__name]
        except KeyError:
            return None

    def __setattr__(self, __name: str, __value: Tensor | Any) -> None:
        self[__name] = __value

    def __setitem__(self, __key: str, __value: Tensor | Any) -> None:
        if isinstance(__value, Tensor):
            if self.shape is None:
                super().__setitem__(__key, __value)
                self.update_shape()
                super().__setattr__("device", __value.device)
                return
            self._validate_shape(self.shape, {__key: __value})
            self._validate_device(self.device, {__key: __value})
        super().__setitem__(__key, __value)

    def _get_data_dim(self, key: str = None, t: Any = None) -> int:
        if key is not None:  # Called from outside with argument `key`
            if key not in self.__annotations__:
                return -1  # Auto dims if not specified in annotations
            t = self.__annotations__[key]
        if isinstance(t, UnionType):
            return max([self._get_data_dim(t=t1) for t1 in t.__args__])
        if isclass(t):
            if issubclass(t, _Tensor):
                return t.data_dim
            elif issubclass(t, Tensor):
                return -1  # Auto dims no data_dim is specified
        return -2  # t is not `Tensor` type

    def _get_batch_shape(self, tensor: Tensor, data_dim: int):
        return list(tensor.shape) if data_dim == 0 else list(tensor.shape[:-data_dim])

    def _infer_shape_from_tensors_only(self, tensors: dict[str, Tensor]):
        """
        Infer `shape` from input tensors without relying on type annotations.

        Called when no type annotations are available for tensors.

        :param tensors `{str: Tensor}`: tensors
        """
        if len(tensors) == 0:  # Input contains no tensors
            # `shape` is empty
            super().__setattr__("shape", None)
        elif len(tensors) == 1:  # Input contains only one tensor
            # `shape` equals to the tensor's shape
            super().__setattr__("shape", list(next(tensors.values().__iter__()).shape))
        else:  # Input contains more than one tensor
            # Get same leading shapes in input tensors
            first_tensor = next(tensors.values().__iter__())
            shape = []
            for i, s in enumerate(first_tensor.shape):
                if countOf([tensor.shape[i] for tensor in tensors.values()
                            if len(tensor.shape) > i], s) == len(tensors):
                    shape.append(s)
                else:
                    break
            super().__setattr__("shape", shape)

    def _validate_device(self, device: torch.device, tensors: dict[str, Tensor]):
        """
        Validate whether specified tensors match the device.

        :param device `torch.device`: the required device
        :param tensors `{str: Tensor}`: tensors to validate
        :raises ValueError: if some tensor doesn't match the device
        """
        for key, tensor in tensors.items():
            if tensor.device != device:
                raise ValueError(f"Require \"{key}\" on {device} but on {tensor.device}")

    def _validate_shape(self, shape: list[int], tensors: dict[str, Tensor]):
        """
        Validate whether specified tensors match the batch shape.

        :param shape `[int]`: the required batch shape
        :param tensors `{str: Tensor}`: tensors to validate
        :raises ValueError: if some tensor doesn't match the shape
        """
        # Validate whether all tensors in data match the batch shape and device
        for key, val in tensors.items():
            data_dim = self._get_data_dim(key)
            match(data_dim):
                case -2:
                    raise ValueError(f"\"{key}\" cannot be a Tensor")
                case -1:
                    if list(val.shape[:len(shape)]) != shape:
                        raise ValueError(f"\"{key}\" does not match the batch shape {shape}")
                case _:
                    if self._get_batch_shape(val, data_dim) != shape:
                        raise ValueError(f"\"{key}\" does not match the batch shape {shape}")
