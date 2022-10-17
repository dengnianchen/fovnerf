from operator import itemgetter
from configargparse import ArgumentParser

from utils import device, netio
from utils.args import BaseArgs
from utils.types import *
from model import Model
from data import Dataset


trainer_classes: dict[str, Type["Trainer"]] = {}


class Trainer:
    class Args(BaseArgs):
        trainset: Path

    args: Args
    states: dict[str, Any]

    @staticmethod
    def get_class(name: str) -> Type["Trainer"] | None:
        typename = name if name.endswith("Trainer") else f"{name}Trainer"
        return trainer_classes.get(typename)

    def state_dict(self) -> dict[str, Any]:
        return {
            "model": self.model.state_dict(),
            "epoch": self.epoch,
            "iters": self.iters,
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict() if self.scheduler else None
        }

    def load_state_dict(self, state_dict: dict[str, Any]):
        self.epoch = state_dict.get("epoch", self.epoch)
        self.iters = state_dict.get("iters", self.iters)
        if "model" in state_dict:
            self.model.load_state_dict(state_dict["model"])
        if "optimizer" in state_dict:
            self.optimizer.load_state_dict(state_dict["optimizer"])
        if self.scheduler and "scheduler" in state_dict:
            self.scheduler.load_state_dict(state_dict["scheduler"])

    def train(self):
        raise NotImplementedError

    @staticmethod
    def start_new_training(exp_name: str = None, config_path: Path = None):
        print(f"Start new training >>>")
        if config_path:
            if not config_path.exists():
                raise FileNotFoundError(f"{config_path} is not found.")
            print(f"Config file: {config_path}")

        # First parse model class and trainer class from config file or command-line arguments
        parser = ArgumentParser(default_config_files=[f"{config_path}"] if config_path else [])
        parser.add_argument('--model', type=str, required=True,
                            help='The model to train')
        parser.add_argument('--trainer', type=str, default="Basic",
                            help='The trainer to use for training')
        args = parser.parse_known_args()[0]

        # Parse trainer's args
        TrainerCls = Trainer.get_class(args.trainer)
        trainer_args = TrainerCls.Args().parse(config_path)

        # Load training dataset
        trainset = Trainer._load_dataset(trainer_args.trainset)

        # Setup model
        # Note: Some model's args are inferred from training dataset
        ModelCls = Model.get_class(args.model)
        model_args = ModelCls.Args()
        if trainset.depth_range:
            model_args.near = trainset.depth_range[0]
            model_args.far = trainset.depth_range[1]
        model_args.parse(config_path)
        model = ModelCls(model_args).to(device.default())

        # Start training
        run_dir = trainset.root / "_nets" / trainset.name / exp_name
        run_dir.mkdir(parents=True, exist_ok=True)
        TrainerCls(model, run_dir, trainer_args).train(trainset)

    @staticmethod
    def continue_training(ckpt_path: Path):
        ckpt = netio.load_checkpoint(ckpt_path)
        args, states = itemgetter("args", "states")(ckpt)
        print(f"Continue training from checkpoint {ckpt['path']} >>>")

        # Setup model
        ModelCls = Model.get_class(args["model"])
        model_args = ModelCls.Args(**args["model_args"])
        model = ModelCls(model_args).to(device.default())

        # Setup trainer
        # Note: arguments for trainer can be overriden by command-line arguments
        TrainerCls = Trainer.get_class(args["trainer"])
        trainer_args = TrainerCls.Args(**args["trainer_args"]).parse()
        trainer = TrainerCls(model, ckpt["path"].parent, trainer_args)

        trainset = Trainer._load_dataset(trainer_args.trainset)
        trainer.load_state_dict(states)
        trainer.train(trainset)

    @staticmethod
    def _load_dataset(data_path: Path) -> Dataset:
        dataset = Dataset(data_path)
        print(f"Dataset: {dataset.root}/{dataset.name}")
        return dataset
