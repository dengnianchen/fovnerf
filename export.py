import os
import argparse
import torch
from pathlib import Path

from utils import netio, device
from model import Model


parser = argparse.ArgumentParser()
parser.add_argument('--batch-size', type=eval, required=True,
                    help='The batch size of exported model. A python expression that result in integer is acceptable')
parser.add_argument('--outdir', type=str, default='exported',
                    help='The output directory under the directory of model, defaults to "exported"')
parser.add_argument('--trt', action="store_true",
                    help='If specified, convert exported ONNX model to TensorRT workspace using trtexec')
parser.add_argument('model', type=Path,
                    help='Model to export')
args = parser.parse_args()

with torch.inference_mode():
    ckpt_path = netio.find_checkpoint(args.model)
    if ckpt_path is None:
        raise FileNotFoundError(f"{ckpt_path} does not contain checkpoints")
    out_dir: Path = ckpt_path.parent / args.outdir
    out_dir.mkdir(exist_ok=True)
    onnx_paths = Model.load(ckpt_path, device.default(), True).export(out_dir, ckpt_path.stem,
                                                                      args.batch_size)
    if args.trt:
        for onnx_path in onnx_paths:
            os.system(f"trtexec --onnx={onnx_path} --fp16 --saveEngine={onnx_path.with_suffix('.trt')} "
                      "--workspace=8192 --noDataTransfers")
    print(f'Model exported to {out_dir}')
