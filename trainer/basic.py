import torch
from collections import defaultdict
from statistics import mean
from tqdm import tqdm
from tensorboardX import SummaryWriter
from torch.optim import Optimizer, Adam
from torch.optim.lr_scheduler import _LRScheduler, ExponentialLR

from .trainer import Trainer
from utils import netio, logging, metrics, misc
from utils.types import *
from data import Dataset, RaysLoader
from model import Model


class BasicTrainer(Trainer):
    class Args(Trainer.Args):
        max_iters: int | None
        max_epochs: int = 20
        checkpoint_interval: int | None
        batch_size: int = 4096
        loss: list[str] = ["Color_L2"]
        lr: float = 5e-4
        lr_decay: float | None

    args: Args
    states: dict[str, Any]
    optimizer: Optimizer
    scheduler: _LRScheduler | None
    loss_defs = {
        "Color_L2": {
            "fn": lambda out, gt: metrics.mse(out["color"], gt["color"]),
            "required_outputs": ["color"]
        },
        "CoarseColor_L2": {
            "fn": lambda out, gt: metrics.mse(out["coarse_color"], gt["color"]),
            "required_outputs": ["coarse_color"]
        }
    }

    def __init__(self, model: Model, run_dir: Path, args: Args = None) -> None:
        self.model = model.train()
        self.run_dir = run_dir
        self.args = args or self.__class__.Args()
        self.epoch = 0
        self.iters = 0

        if self.args.max_iters:  # iters mode
            self.max_iters = self.args.max_iters
            self.max_epochs = None
            self.checkpoint_interval = self.args.checkpoint_interval or 10000
        else:  # epochs mode
            self.max_iters = None
            self.max_epochs = self.args.max_epochs
            self.checkpoint_interval = self.args.checkpoint_interval or 10

        self._init_optimizer()
        self._init_scheduler()

        self.required_outputs = []
        for key in self.args.loss:
            self.required_outputs += self.loss_defs[key]["required_outputs"]
        self.required_outputs = list(set(self.required_outputs))

        tb_log_dir = self.run_dir / "_log"
        tb_log_dir.mkdir(exist_ok=True)
        self.tb_writer = SummaryWriter(tb_log_dir, purge_step=0)
        logging.initialize(self.run_dir / "train.log")

        logging.print_and_log(f"Model arguments: {self.model.args}")
        logging.print_and_log(f"Trainer arguments: {self.args}")

        # Debug: print model structure
        print(model)

    def reset_optimizer(self):
        self._init_optimizer()
        if self.scheduler is not None:
            scheduler_state = self.scheduler.state_dict()
            self._init_scheduler()
            self.scheduler.load_state_dict(scheduler_state)

    def train(self, dataset: Dataset):
        self.rays_loader = RaysLoader(dataset, self.args.batch_size, shuffle=True,
                                      device=self.model.device)
        self.forward_chunk_size = self.args.batch_size

        if self.max_iters:
            print(f"Begin training... Max iters: {self.max_iters}")
            self.progress = tqdm(total=self.max_iters, dynamic_ncols=True)
            self.rays_iter = self.rays_loader.__iter__()
            while self.iters < self.max_iters:
                self._train_iters(min(self.checkpoint_interval, self.max_iters - self.iters))
                self._save_checkpoint()
        else:
            print(f"Begin training... Max epochs: {self.max_epochs}")
            while self.epoch < self.max_epochs:
                self._train_epoch()
                self._save_checkpoint()

        print("Train finished")

    def _init_scheduler(self):
        self.scheduler = self.args.lr_decay and ExponentialLR(self.optimizer, self.args.lr_decay)

    def _init_optimizer(self):
        self.optimizer = Adam(self.model.parameters(), lr=self.args.lr)

    def _save_checkpoint(self):
        if self.checkpoint_interval is None:
            return

        ckpt = {
            "args": {
                "model": self.model.__class__.__name__,
                "model_args": vars(self.model.args),
                "trainer": self.__class__.__name__,
                "trainer_args": vars(self.args)
            },
            "states": self.state_dict()
        }

        if self.max_iters:
            # For iters mode, a checkpoint will be saved every `checkpoint_interval` iterations
            netio.save_checkpoint(ckpt, self.run_dir, self.iters)
        else:
            # For epochs mode, a checkpoint will be saved every epoch.
            # Checkpoints which don't match `checkpoint_interval` will be cleaned later
            netio.clean_checkpoints(self.run_dir, self.checkpoint_interval)
            netio.save_checkpoint(ckpt, self.run_dir, self.epoch)

    def _update_progress(self, loss: float = 0):
        self.progress.set_postfix_str(f"Loss: {loss:.2e}" if loss > 0 else "")
        self.progress.update()

    def _forward(self, rays: Rays) -> RenderOut:
        return self.model(rays, *self.required_outputs)

    def _compute_loss(self, rays: Rays, out: RenderOut) -> dict[str, Tensor]:
        loss_terms: dict[str, Tensor] = {}
        for key in self.args.loss:
            try:
                loss_terms[key] = self.loss_defs[key]["fn"](out, rays)
            except KeyError:
                pass
        # Debug: print loss terms
        #self.progress.write(",".join([f"{key}: {value.item():.2e}" for key, value in loss_terms.items()]))
        return loss_terms

    def _train_iter(self, rays: Rays) -> float:
        try:
            self.optimizer.zero_grad(True)
            loss_terms = defaultdict(list)
            for offset in range(0, rays.shape[0], self.forward_chunk_size):
                rays_chunk = rays.select(slice(offset, offset + self.forward_chunk_size))
                out_chunk = self._forward(rays_chunk)
                loss_chunk = self._compute_loss(rays_chunk, out_chunk)
                loss_value = sum(loss_chunk.values())
                loss_value.backward()
                loss_terms["Overall_Loss"].append(loss_value.item())
                for key, value in loss_chunk.items():
                    loss_terms[key].append(value.item())
            loss_terms = {key: mean(value) for key, value in loss_terms.items()}
            
            self.optimizer.step()
            if self.scheduler:
                self.scheduler.step()
            self.iters += 1

            if hasattr(self, "tb_writer"):
                for key, value in loss_terms.items():
                    self.tb_writer.add_scalar(f"Loss/{key}", value, self.iters)

            return loss_terms["Overall_Loss"]
        except RuntimeError as e:
            if not e.__str__().startswith("CUDA out of memory"):
                raise e
        self.progress.write("CUDA out of memory, half forward batch and retry.")
        logging.warning("CUDA out of memory, half forward batch and retry.")
        self.forward_chunk_size //= 2
        torch.cuda.empty_cache()
        return self._train_iter(rays)

    def _train_iters(self, iters: int):
        recent_loss_list = []
        tot_loss = 0
        start_event = torch.cuda.Event(True)
        end_event = torch.cuda.Event(True)
        start_event.record()
        for _ in range(iters):
            try:
                rays = self.rays_iter.__next__()
            except StopIteration:
                self.rays_iter = self.rays_loader.__iter__()  # A new epoch
                rays = self.rays_iter.__next__()
            loss_val = self._train_iter(rays)
            recent_loss_list = (recent_loss_list + [loss_val])[-50:]  # Keep recent 50 iterations
            recent_avg_loss = sum(recent_loss_list) / len(recent_loss_list)
            tot_loss += loss_val
            self._update_progress(recent_avg_loss)
        end_event.record()
        torch.cuda.synchronize()
        avg_time = start_event.elapsed_time(end_event) / 1000 / iters
        avg_loss = tot_loss / iters
        state_str = f"Iter {self.iters}: Avg. {misc.format_time(avg_time)}/iter; Loss: {avg_loss:.2e}"
        self.progress.write(state_str)
        logging.info(state_str)

    def _train_epoch(self):
        iters_per_epoch = len(self.rays_loader)
        recent_loss_list = []
        tot_loss = 0

        self.progress = tqdm(total=iters_per_epoch, desc=f"Epoch {self.epoch + 1:<3d}", leave=False,
                             dynamic_ncols=True)
        start_event = torch.cuda.Event(True)
        end_event = torch.cuda.Event(True)
        start_event.record()
        for rays in self.rays_loader:
            loss_val = self._train_iter(rays)
            recent_loss_list = (recent_loss_list + [loss_val])[-50:]
            recent_avg_loss = sum(recent_loss_list) / len(recent_loss_list)
            tot_loss += loss_val
            self._update_progress(recent_avg_loss)
        self.progress.close()
        end_event.record()
        torch.cuda.synchronize()
        self.epoch += 1
        epoch_time = start_event.elapsed_time(end_event) / 1000
        avg_time = epoch_time / iters_per_epoch
        avg_loss = tot_loss / iters_per_epoch
        state_str = f"Epoch {self.epoch} spent {misc.format_time(epoch_time)} "\
            f"(Avg. {misc.format_time(avg_time)}/iter). Loss is {avg_loss:.2e}."
        logging.print_and_log(state_str)
