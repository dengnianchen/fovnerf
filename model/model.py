from .__common__ import *
from torch import onnx
from operator import itemgetter
from utils import netio
from utils.args import BaseArgs


model_classes: dict[str, "Model"] = {}


class Model(nn.Module):
    class Args(BaseArgs):
        ...

    args: Args

    def __init__(self, args: Args):
        super().__init__()
        self.args = args

    def __call__(self, rays: Rays, *outputs: str, **args) -> RenderOut:
        raise NotImplementedError()

    def export(self, out_dir: Path, name: str, batch_size: int) -> list[Path]:
        raise NotImplementedError()

    @staticmethod
    def get_class(typename: str) -> Type["Model"] | None:
        return model_classes.get(typename)

    @staticmethod
    def create(typename: str, args: dict | Args) -> "Model":
        ModelCls = Model.get_class(typename)
        if ModelCls is None:
            raise ValueError(f"Model {typename} is not found")
        if isinstance(args, dict):
            args = ModelCls.Args(**args)
        return ModelCls(args)

    @staticmethod
    def load(path: Path, device: torch.device = None, eval_mode: bool = False) -> "Model":
        ckpt = netio.load_checkpoint(path)
        model_type, model_args = itemgetter("model", "model_args")(ckpt["args"])
        model = Model.create(model_type, model_args)
        model.load_state_dict(ckpt["states"]["model"])
        if device:
            model.to(device)
        if eval_mode:
            model.eval()
        return model

    def export_onnx(model: nn.Module, path: Path, input_shapes: dict[str, list[int]],
                    output_names: list[str]):
        input_tensors = tuple(
            torch.empty(shape, device=model.device)
            for shape in input_shapes.values()
        )
        onnx.export(model, input_tensors, path,
                    export_params=True,  # store the trained parameter weights inside the model file
                    verbose=True,
                    opset_version=9,     # the ONNX version to export the model to
                    do_constant_folding=True,  # whether to execute constant folding
                    input_names=list(input_shapes.keys()),   # the model's input names
                    output_names=output_names)  # the model's output names
