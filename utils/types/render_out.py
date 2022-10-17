from .__common__ import *
from .data_pack import *


class RenderOut(DataPack):
    color: Tensor1D | None
    depth: Tensor1D | None
    colors: Tensor2D | None
    densities: Tensor2D | None
    alphas: Tensor2D | None
    weights: Tensor2D | None
