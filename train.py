from configargparse import ArgumentParser
from pathlib import Path

from trainer import Trainer


initial_parser = ArgumentParser()
initial_parser.add_argument('-c', '--config', type=str,
                            help='Config name, ignored if argument "--ckpt" is specified')
initial_parser.add_argument('-e', '--expname', type=str,
                            help='Experiment name, defaults to config name, ignored if argument "--ckpt" is specified')
initial_parser.add_argument('-p', '--ckpt', type=Path,
                            help='Path to checkpoint file')
init_args = initial_parser.parse_known_args()[0]

root_dir = Path(__file__).absolute().parent

if init_args.ckpt:
    Trainer.continue_training(init_args.ckpt)
else:
    Trainer.start_new_training(
        init_args.expname or init_args.config or "unnamed",
        init_args.config and root_dir / "configs" / f"{init_args.config}.ini"
    )
