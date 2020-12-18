from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, Any, Tuple, Union

from colliflow.typing import Dtype, JsonDict, Shape

if TYPE_CHECKING:
    from colliflow.modules.module import Module


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
        return f"Tensor(shape={self.shape}, dtype={self.dtype})"


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
