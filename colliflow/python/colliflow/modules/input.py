from colliflow.tensors import SymbolicTensor, Tensor

from .module import Module


def Input(shape, dtype, scheduler=None):  # pylint: disable=invalid-name
    x = SymbolicTensor(shape, dtype)
    return InputLayer(shape, dtype, scheduler=scheduler)(x)


class InputLayer(Module):
    name = "Input"

    def __init__(self, shape, dtype, **kwargs):
        super().__init__(shape, dtype, **kwargs)

    def inner_config(self):
        return {"shape": self.shape, "dtype": self.dtype}

    def forward(self, tensor: Tensor):
        return tensor
