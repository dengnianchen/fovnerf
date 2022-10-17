from operator import countOf
from types import UnionType
from typing_extensions import Self
from configargparse import ArgumentParser, Namespace

from .types import *


class BaseArgs(Namespace):

    @property
    def defaults(self) -> dict[str, Any]:
        return {
            key: getattr(self.__class__, key)
            for key in self.__annotations__ if hasattr(self.__class__, key)
        }
    
    def __init__(self, **kwargs) -> None:
        super().__init__(**self.defaults | kwargs)

    def merge_with(self, dict: dict[str, Any]) -> Self:
        return self.__class__(**vars(self) | dict)

    def parse(self, config_path: Path = None, debug: bool = False) -> Self:
        parser = ArgumentParser(default_config_files=[str(config_path)] if config_path else [])
        self.setup_parser(parser, debug)
        return parser.parse_known_args(namespace=self)[0]

    def setup_parser(self, parser: ArgumentParser, debug: bool = False):
        def build_debug_str(key: str, params_for_parser: dict[str, Any], prefix="parser") -> str:
            def to_str(value): return value.__name__ if isinstance(value, Type) else (
                f"\"{value}\"" if isinstance(value, str) else value.__str__())
            params_str = ", ".join([
                f"{name}={to_str(value)}" for name, value in params_for_parser.items()
            ])
            return f"{prefix}.add_argument(\"--{key}\", {params_str})"

        def add_argument(parser: ArgumentParser, key: str, type: Type, required: bool, **kwargs):
            params = {}
            if type == bool:
                bool_group = parser.add_mutually_exclusive_group()
                bool_group.add_argument(f"--{key}", action="store_true")
                bool_group.add_argument(f"--no-{key}", action="store_false", dest=key)
                if debug:
                    print("bool_group = parser.add_mutually_exclusive_group()")
                    print(build_debug_str(key, {"action": "store_true"}, "bool_group"))
                    print(build_debug_str(f"no-{key}", {"action": "store_false", "dest": key},
                                          "bool_group"))
            else:
                params["type"] = type
                if "nargs" in kwargs:
                    params["nargs"] = kwargs["nargs"]
                if "default" in kwargs:
                    params["default"] = kwargs["default"]
                elif required:
                    params["required"] = True
                parser.add_argument(f"--{key}", **params)
                if debug:
                    print(build_debug_str(key, params))

        for key, arg_type in self._get_annotations().items():
            required = True
            kwargs = {}
            if isinstance(arg_type, UnionType):
                if len(arg_type.__args__) != 2 or countOf(arg_type.__args__, type(None)) != 1:
                    raise ValueError(f"{key} cannot be union of two or more different types")
                arg_type = arg_type.__args__[0] if arg_type.__args__[1] == type(None) \
                    else arg_type.__args__[1]
                required = False
            if getattr(arg_type, "__origin__", None) == list:
                arg_type = arg_type.__args__[0]
                kwargs["nargs"] = "*"
            elif getattr(arg_type, "__origin__", None) == tuple:
                arg_type = arg_type.__args__[0]
                if any([arg != arg_type for arg in arg_type.__args__]):
                    raise ValueError(f"{key} cannot be tuple of different types")
                kwargs["nargs"] = len(arg_type.__args__)
            if hasattr(self, key):
                kwargs["default"] = getattr(self, key)
            add_argument(parser, key, arg_type, required, **kwargs)

    @classmethod
    def _get_annotations(type: Type["BaseArgs"]) -> dict[str, Any]:
        if type == BaseArgs:
            return type.__annotations__
        return type.__base__._get_annotations() | type.__annotations__

