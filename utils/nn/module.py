import torch
from collections import OrderedDict
from typing import Any


class ModuleMeta(type):
    """
    This meta class aims at changing module method `__call__` to `forward`.
    
    Because PyTorch requires a module implements its forward process in `forward` method but called 
    in `__call__` form, the IDE fails to provide type hint for it. With this meta class, the forward 
    process can be defined in `__call__` and make the type hint work.
    """

    def __new__(cls, name, bases, attrs):
        """
        New a module class and change method `__call__` to `forward`
        """
        if "__call__" in attrs:
            attrs["forward"] = attrs["__call__"]
            del attrs["__call__"]
        return type.__new__(cls, name, bases, attrs)


class Module(torch.nn.Module, metaclass=ModuleMeta):
    """
    An extension to PyTorch's Module class which supports:
    * Implementing forward process in __call__ method so that IDE can provide correct type hint in caller;
    * Information about a module's inputs and outputs (name and channels);
    * An addition field `device` to indicate which device a module is on;
    * Stub hook methods for before and after `load_state_dict`, which can be override in subclass.
    """

    @property
    def in_chns(self) -> int | dict[str, int]:
        """`int|{str:int}` Channels of input(s). Return `int` if the module has only one input."""
        match(len(self._in_chns)):
            case 0:
                return 0
            case 1:
                return next(self._in_chns.values().__iter__())
            case _:
                return self._in_chns

    @property
    def out_chns(self) -> int | dict[str, int]:
        """`int|{str:int}` Channels of output(s). Return `int` if the module has only one output."""
        match(len(self._out_chns)):
            case 0:
                return 0
            case 1:
                return next(self._out_chns.values().__iter__())
            case _:
                return self._out_chns

    def __init__(self, in_chns: dict[str, int] = None, out_chns: dict[str, int] = None):
        """
        Initialize a module.

        :param in_chns `{str:int}?`: channels of inputs, defaults to None
        :param out_chns `{str:int}?`: channels of outputs, defaults to None
        """
        super().__init__()
        self.device = torch.device("cpu")
        self._in_chns = in_chns or {}
        self._out_chns = out_chns or {}
        self._temp = OrderedDict()
        self._register_load_state_dict_pre_hook(self._before_load_state_dict)

    def chns(self, name: str) -> int:
        """
        Get channels of specified input/output.

        :param name `str`: the name of specified input/output with prefix "in" or "out".
            e.g. "inpos" is for the input named "pos"
        :return `int`: number of channels
        """
        if name.startswith("in"):
            return self._in_chns.get(name[2:], 0)
        if name.startswith("out"):
            return self._out_chns.get(name[3:], 0)
        return self._in_chns.get(name, self._out_chns.get(name, 0))

    # Override: to ensure the newly added submodule is on the correct device and `device` field has the correct value
    def add_module(self, name: str, module: torch.nn.Module | None) -> None:
        if isinstance(module, torch.nn.Module):
            module = module.to(self.device)
        return super().add_module(name, module)

    # Override: to ensure the newly added parameter is on the correct device
    def register_parameter(self, name: str, param: torch.nn.Parameter | None) -> None:
        if isinstance(param, torch.nn.Parameter):
            param = param.to(self.device)
        return super().register_parameter(name, param)

    # Override: to ensure the newly added buffer is on the correct device
    def register_buffer(self, name: str, tensor: torch.Tensor | None, persistent: bool = True) -> None:
        if isinstance(tensor, torch.Tensor):
            tensor = tensor.to(self.device)
        return super().register_buffer(name, tensor, persistent=persistent)

    # Override: to update `device` field
    def to(self, *args, **kwargs):
        super().to(*args, **kwargs)
        target_device = None
        try:
            target_device = torch.device(kwargs['device'] if 'device' in kwargs else args[0])
        except Exception:
            pass
        # Update the `device` field of self and submodules
        if target_device is not None:
            def move_to_device(m):
                if isinstance(m, Module):
                    m.device = target_device
            self.apply(move_to_device)
        return self

    # Override: to support `_after_load_state_dict` hook
    def load_state_dict(self, state_dict: 'OrderedDict[str, torch.Tensor]', strict: bool = True):
        ret = super().load_state_dict(state_dict, strict=strict)

        def fn(module):
            if isinstance(module, Module):
                module._after_load_state_dict()
        self.apply(fn)
        return ret

    # Override: to ensure the value to set is on the correct device and support setting temp child
    def __setattr__(self, name: str, value: torch.Tensor | torch.nn.Module | Any) -> None:
        if isinstance(value, (torch.Tensor, torch.nn.Module)):
            value = value.to(self.device)
        super().__setattr__(name, value)

    def _before_load_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys,
                                unexpected_keys, error_msgs):
        """
        This hook will be called before loading the state dict.
        """
        pass

    def _after_load_state_dict(self) -> None:
        """
        This hook will be called after the state dict is loaded.
        """
        pass
