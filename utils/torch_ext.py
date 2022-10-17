import torch

from .misc import calculate_autosize


def grid2d(rows: int, cols: int = None, normalize: bool = False, indexing: str = "xy",
           device: torch.device = None) -> torch.Tensor:
    """
    Generate a 2D grid

    :param rows `int`: number of rows
    :param cols `int`: number of columns
    :param normalize `bool`: whether return coords in normalized space, defaults to False
    :param indexing `str`: specify the order of returned coordinates. Optional values are "xy" and "ij",
        defaults to "xy"
    :return `Tensor(R, C, 2)`: the coordinates of the grid
    """
    if cols is None:
        cols = rows
    i, j = torch.meshgrid(torch.arange(rows, device=device),
                          torch.arange(cols, device=device), indexing="ij")  # (R, C)
    if normalize:
        i.div_(rows - 1)
        j.div_(cols - 1)
    return torch.stack([j, i] if indexing == "xy" else [i, j], 2)  # (R, C, 2)


def union(*tensors: torch.Tensor | float) -> torch.Tensor:
    try:
        first_tensor = next((item for item in tensors if isinstance(item, torch.Tensor)))
    except StopIteration:
        raise ValueError("Arguments should contain at least one tensor")
    tensors = [
        item if isinstance(item, torch.Tensor) else first_tensor.new_tensor([item])
        for item in tensors
    ]
    if any(item.device != first_tensor.device or item.dtype != first_tensor.dtype
           for item in tensors):
        raise ValueError("All tensors should have same dtype and locate on same device")
    shape = torch.broadcast_shapes(*(item.shape[:-1] for item in tensors))
    return torch.cat([item.expand(*shape, -1) for item in tensors], dim=-1)


def split(tensor: torch.Tensor, *sizes: int) -> tuple[torch.Tensor, ...]:
    
    sizes, tot_size = calculate_autosize(tensor.shape[-1], *sizes)
    if tot_size < tensor.shape[-1]:
        sizes = [*sizes, tensor.shape[-1] - tot_size]
        return torch.split(tensor, sizes, -1)[:-1]
    else:
        return torch.split(tensor, sizes, -1)
