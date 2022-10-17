from .__common__ import *
from pathlib import Path


setattr(Path, "__mod__", lambda self, indices: MultiPaths(str(self), indices))


class MultiPaths(Iterable[Path]):

    class Iter(Iterator[Path]):
        def __init__(self, container: "MultiPaths") -> None:
            super().__init__()
            self.path_pattern = container.path_pattern
            self.index_iter = container.indices.__iter__()

        def __next__(self) -> Path:
            index = self.index_iter.__next__()
            return Path(self.path_pattern % index)

    def __init__(self, path_pattern: str, indices: int | Iterable[int] | range):
        super().__init__()
        self.path_pattern = path_pattern
        self.indices = [indices] if isinstance(indices, int) else indices

    def __iter__(self) -> Iter:
        return MultiPaths.Iter(self)