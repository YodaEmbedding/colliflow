from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, Any, Tuple, Union

import numpy as np

from colliflow.typing import Dtype, JsonDict, Shape

if TYPE_CHECKING:
    from colliflow.modules.module import (  # pylint: disable=cyclic-import
        Module,
    )


@dataclass
class TensorInfo:
    dtype: Dtype
    shape: Shape

    @staticmethod
    def from_(x: Union["TensorInfo", JsonDict]):
        if isinstance(x, TensorInfo):
            return TensorInfo(**asdict(x))
        if isinstance(x, dict):
            return TensorInfo(**x)
        raise NotImplementedError

    def as_dict(self):
        return asdict(self)


@dataclass
class Tensor:
    shape: Shape
    dtype: Dtype
    data: Any

    def __post_init__(self):
        if self.shape is None or self.dtype is None:
            raise ValueError("Please ensure shape and dtype are correct.")

    def __repr__(self) -> str:
        data = self.data
        return f"Tensor(shape={self.shape}, dtype={self.dtype}, data={data})"

    def __eq__(self, other: "Tensor"):
        def is_none_shape(x):
            return len(x.shape) == 1 and x.shape[0] is None

        dtype_eq = self.dtype == other.dtype
        shape_eq = (
            is_none_shape(self) and is_none_shape(other)
        ) or self.shape == other.shape
        data_eq = (
            np.all(self.data == other.data)
            if isinstance(self, np.ndarray)
            else self.data == other.data
        )

        return dtype_eq and shape_eq and data_eq


@dataclass
class SymbolicTensor:
    shape: Shape
    dtype: Dtype
    parent: "Module" = None

    def __post_init__(self):
        if self.shape is None or self.dtype is None:
            raise ValueError("Please ensure shape and dtype are correct.")

    def __repr__(self) -> str:
        s = f"shape={self.shape}, dtype={self.dtype}, parent={self.parent!r}"
        return f"SymbolicTensor({s})"

    @property
    def config(self) -> Tuple[Shape, Dtype]:
        return self.shape, self.dtype


__all__ = [
    "Dtype",
    "Shape",
    "SymbolicTensor",
    "Tensor",
    "TensorInfo",
]
