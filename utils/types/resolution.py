from .__common__ import NamedTuple


class Resolution(NamedTuple):
    h: int
    w: int

    @staticmethod
    def from_str(res_str: str) -> "Resolution":
        arr = res_str.split("x")
        return Resolution(int(arr[1]), int(arr[0]))

    def __repr__(self) -> str:
        return f"{self.w}x{self.h}"