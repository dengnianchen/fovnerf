from .weight_init import *
from .module import Module


class FcLayer(Module):
    """
    This module is a wrap of torch's Linear module, adding support for activation function and 
    layer normalization.
    """

    def __init__(self, in_chns: int, out_chns: int, act: str = 'linear', with_ln: bool = False):
        """
        Initialize a full-connection layer module. 

        :param in_chns `int`: channels of input
        :param out_chns `int`: channels of output
        :param act `str?`: the activation function, defaults to `"linear"`
        :param with_ln `bool?`: whether to apply layer normalization to the output, defaults to `False`
        """
        super().__init__({"x": in_chns}, {"y": out_chns})
        nls_and_inits = {
            'relu': (nn.ReLU, init_weights_relu),
            'leakyrelu': (nn.LeakyReLU, init_weights_leakyrelu),
            'sigmoid': (nn.Sigmoid, init_weights_xavier),
            'tanh': (nn.Tanh, init_weights_xavier),
            'selu': (nn.SELU, init_weights_selu),
            'softplus': (nn.Softplus, init_weights_trunc_normal),
            'elu': (nn.ELU, init_weights_elu),
            'softmax': (nn.Softmax, init_weights_softmax),
            'logsoftmax': (nn.LogSoftmax, init_weights_softmax),
            'linear': (nn.Identity, init_weights_xavier)
        }
        nl_cls, weight_init_fn = nls_and_inits[act]

        self.net = [nn.Linear(in_chns, out_chns)]
        if with_ln:
            self.net.append(nn.LayerNorm([out_chns]))
        self.net.append(nl_cls())
        self.net = nn.Sequential(*self.net)
        self.net.apply(weight_init_fn)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def __repr__(self) -> str:
        s = f"{self.net[0].in_features} -> {self.net[0].out_features}, "\
            + ", ".join(module.__class__.__name__ for module in self.net[1:])
        return f"{self._get_name()}({s})"


class FcBlock(Module):
    """
    A full-connection block module consists of multiple full-connection layers.
    """

    def __init__(self, in_chns: int, out_chns: int, nl: int, nf: int, skips: list[int] = [],
                 act: str = 'relu', out_act: str = 'linear', with_ln: bool = False):
        """
        Initialize a full-connection block module.

        :param in_chns `int`: channels of input
        :param out_chns `int`: channels of output, if non-zero, an output layer (`nf` -> `out_chns`)
            will be appended, so the block will have `n_layers + 1` layers
        :param nl `int`: number of hidden layers
        :param nf `int`: number of features in each hidden layer
        :param skips `[int]?`: create skip connections from input to hidden layers in this list, defaults to `[]`
        :param act `str?`: the activation function for hidden layers, defaults to `"relu"`
        :param out_act `str?`: the activation function for the output layer defaults to `"linear"`
        :param with_ln `bool?`: whether to apply the layer normalization to each hidden layer's output,
            defaults to `False`
        """
        super().__init__({"x": in_chns}, {"y": out_chns})
        self.skips = skips
        self.layers = nn.ModuleList([
            FcLayer(in_chns, nf, act, with_ln=with_ln)] + [
            FcLayer(nf + (i in skips) * in_chns, nf, act, with_ln=with_ln)
            for i in range(nl - 1)
        ])
        if out_chns:
            self.layers.append(FcLayer(nf, out_chns, out_act, with_ln=False))

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        h = x
        for i, layer in enumerate(self.layers):
            h = layer(h)
            if i in self.skips:
                h = torch.cat([x, h], -1)
        return h

    def __repr__(self) -> str:
        lines = []
        for i, layer in enumerate(self.layers):
            mod_str = repr(layer)
            mod_str = nn.modules.module._addindent(mod_str, 2)
            if i - 1 in self.skips:
                mod_str += " <-"
            lines.append(f"({i}): {mod_str}")

        main_str = self._get_name() + '('
        if lines:
            main_str += '\n  ' + '\n  '.join(lines) + '\n'
        main_str += ')'
        return main_str
