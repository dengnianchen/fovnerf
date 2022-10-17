from torch import Tensor
from typing import Any, Type, Callable, NamedTuple, Iterable, Iterator, overload

IndexSelector = int | slice | list[int] | Tensor