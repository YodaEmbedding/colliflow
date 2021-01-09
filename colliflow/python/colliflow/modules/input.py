from typing import Tuple

from colliflow.tensors import SymbolicTensor, Tensor
from colliflow.typing import Dtype, Shape

from .module import ForwardModule


class InputModule(ForwardModule):
    """Passes input tensors through without any further processing."""

    name = "Input"

    def __init__(self, shape: Shape, dtype: Dtype):
        super().__init__(shape, dtype)

    def inner_config(self):
        return {"shape": self.input_shapes[0], "dtype": self.input_dtypes[0]}

    def forward(self, *inputs: Tensor) -> Tensor:
        return inputs[0]


def Input(shape: Tuple[int], dtype: str):  # pylint: disable=invalid-name
    """Creates an input module of given shape and dtype."""
    x = SymbolicTensor(shape, dtype)
    return InputModule(shape, dtype)(x)


__all__ = [
    "Input",
    "InputModule",
]
