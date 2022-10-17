import torch
from pathlib import Path


checkpoint_file_prefix = "checkpoint_"
checkpoint_file_suffix = ".tar"


def get_checkpoint_filename(epoch):
    return f"{checkpoint_file_prefix}{epoch}{checkpoint_file_suffix}"


def list_epochs(directory: Path) -> list[int]:
    epoch_list = [
        int(file_path.stem[len(checkpoint_file_prefix):])
        for file_path in directory.glob(get_checkpoint_filename("*"))
    ]
    epoch_list.sort()
    return epoch_list


def find_checkpoint(path: Path) -> Path | None:
    if path.suffix != checkpoint_file_suffix:
        existed_epochs = list_epochs(path)
        return path / get_checkpoint_filename(existed_epochs[-1]) if existed_epochs else None
    return path if path.exists() else None


def load_checkpoint(path: Path) -> dict[str, str | dict]:
    path = find_checkpoint(path)
    if path is None:
        raise FileNotFoundError(f"{path} does not contain checkpoint files")
    return torch.load(path) | {"path": path}


def save_checkpoint(states_dict: dict, directory: Path, epoch: int):
    torch.save(states_dict, Path(directory) / get_checkpoint_filename(epoch))


def clean_checkpoints(directory: Path, keep_interval: int):
    (directory / '_misc').mkdir(exist_ok=True)
    for file in directory.glob(f"{checkpoint_file_prefix}*{checkpoint_file_suffix}"):
        i = int(file.name[len(checkpoint_file_prefix):-len(checkpoint_file_suffix)])
        if i % keep_interval != 0:
            file.rename(directory / "_misc" / file.name)
