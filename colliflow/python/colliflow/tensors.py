class Tensor:  # pylint: disable=too-few-public-methods
    def __init__(self, shape, dtype):
        if shape is None or dtype is None:
            raise ValueError("Please ensure shape and dtype are correct.")

        self.shape = tuple(shape)
        self.dtype = dtype

        # TODO hold some actual data!
        # self.data = data

    def __repr__(self) -> str:
        return f"Tensor(shape={self.shape}, dtype={self.dtype})"


class SymbolicTensor(Tensor):  # pylint: disable=too-few-public-methods
    def __init__(self, shape, dtype, parent=None):
        super().__init__(shape, dtype)
        self.parent = parent

    def __repr__(self) -> str:
        s = f"shape={self.shape}, dtype={self.dtype}, parent={self.parent!r}"
        return f"SymbolicTensor({s})"
