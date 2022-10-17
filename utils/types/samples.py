from .__common__ import *
from .data_pack import *


class Samples(DataPack):
    indices: Tensor1D
    """ Tensor(N[, P], 2)` The unique indices of samples, e.g. (i-th ray, j-th sample)"""

    pts: Tensor1D
    """`Tensor(N[, P], 3)` The positions of samples"""

    dirs: Tensor1D
    """`Tensor(N[, P], 3)` The directions of samples"""

    depths: Tensor0D
    """`Tensor(N[, P])`"""

    dists: Tensor0D
    """`Tensor(N[, P])`"""

    t_vals: Tensor0D
    """`Tensor(N[, P])`"""