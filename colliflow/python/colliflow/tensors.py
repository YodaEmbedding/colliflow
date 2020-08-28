from typing import Sequence, Tuple

Shape = Sequence[int]
Dtype = str


class Tensor:  # pylint: disable=too-few-public-methods
    def __init__(self, shape: Shape, dtype: Dtype):
        if shape is None or dtype is None:
            raise ValueError("Please ensure shape and dtype are correct.")

        self.shape = tuple(shape)
        self.dtype = dtype

        # TODO hold some actual data!? Or is that a separate "DataTensor" type
        # Or perhaps Tensor and SymbolicTensor should inherit TensorLike
        # self.data = data

    def __repr__(self) -> str:
        return f"Tensor(shape={self.shape}, dtype={self.dtype})"


class SymbolicTensor(Tensor):  # pylint: disable=too-few-public-methods
    def __init__(self, shape: Shape, dtype: Dtype, parent=None):
        super().__init__(shape, dtype)
        self.parent = parent

    def __repr__(self) -> str:
        s = f"shape={self.shape}, dtype={self.dtype}, parent={self.parent!r}"
        return f"SymbolicTensor({s})"

    @property
    def config(self) -> Tuple[Shape, Dtype]:
        return self.shape, self.dtype
