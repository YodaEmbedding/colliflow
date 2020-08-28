from typing import Tuple

from colliflow.tensors import SymbolicTensor, Tensor

from .module import InputModule


def Input(
    shape: Tuple[int], dtype: str, scheduler=None
):  # pylint: disable=invalid-name
    x = SymbolicTensor(shape, dtype)
    return InputLayer(shape, dtype, scheduler=scheduler)(x)


class InputLayer(InputModule):
    name = "Input"

    def __init__(self, shape: Tuple[int], dtype: str, **kwargs):
        super().__init__(shape, dtype, **kwargs)

    def inner_config(self):
        return {"shape": self.shape, "dtype": self.dtype}

    def forward(self, tensor: Tensor):
        return tensor
