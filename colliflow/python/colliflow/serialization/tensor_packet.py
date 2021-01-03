from dataclasses import dataclass
from typing import Any, Optional, Tuple, Union, cast

import numpy as np

from colliflow.tensors import Tensor
from colliflow.typing import Dtype, Shape

BufferLike = Union[bytes, bytearray, memoryview]

_ARRAY_LIKE = [
    "int8",
    "int16",
    "int32",
    "int64",
    "uint8",
    "uint16",
    "uint32",
    "uint64",
    "float16",
    "float32",
    "float64",
]


@dataclass
class TensorPacket:
    shape: Shape
    dtype: str
    data: bytes

    def to_bytes(self) -> bytes:
        return b"".join(
            [
                _shape_to_bytes(self.shape),
                _dtype_to_bytes(self.dtype),
                _data_to_bytes(self.data),
            ]
        )

    def to_tensor(self) -> Tensor:
        return Tensor(
            shape=self.shape,
            dtype=self.dtype,
            data=_tensor_data_from_bytes(self.shape, self.dtype, self.data),
        )

    @staticmethod
    def from_bytes(buf: BufferLike) -> Tuple[int, Optional["TensorPacket"]]:
        view = memoryview(buf)
        num_bytes = 0

        try:
            n, shape = _shape_from_bytes(view[num_bytes:])
            num_bytes += n

            n, dtype = _dtype_from_bytes(view[num_bytes:])
            num_bytes += n

            n, data = _data_from_bytes(view[num_bytes:])
            num_bytes += n
        except ParseError:
            return 0, None

        tensor_packet = TensorPacket(shape=shape, dtype=dtype, data=data)

        return num_bytes, tensor_packet

    @staticmethod
    def from_tensor(tensor: Tensor) -> "TensorPacket":
        return TensorPacket(
            shape=tensor.shape,
            dtype=tensor.dtype,
            data=_tensor_data_to_bytes(tensor),
        )


class ParseError(Exception):
    pass


def _shape_to_bytes(shape: Shape) -> bytes:
    """Converts Shape to serialized byte format.

    NOTE: None values are replaced with -1 before serialization.
    """
    shape = tuple(x if x is not None else -1 for x in shape)
    b_len = len(shape).to_bytes(4, byteorder="big")
    b_shape = b"".join(
        x.to_bytes(4, byteorder="big", signed=True) for x in shape
    )
    return b"".join([b_len, b_shape])


def _shape_from_bytes(buf: BufferLike) -> Tuple[int, Shape]:
    if len(buf) < 4:
        raise ParseError
    shape_len = int.from_bytes(buf[:4], byteorder="big")
    if len(buf) < 4 + shape_len:
        raise ParseError
    chunks = [buf[4 * i + 4 : 4 * i + 8] for i in range(shape_len)]
    xs = [int.from_bytes(x, byteorder="big") for x in chunks]
    shape = tuple(x if x != -1 else None for x in xs)
    bytes_read = 4 + 4 * shape_len
    return bytes_read, cast(Shape, shape)


def _dtype_to_bytes(dtype: Dtype) -> bytes:
    b_dtype = dtype.encode()
    b_len = len(b_dtype).to_bytes(4, byteorder="big")
    return b"".join([b_len, b_dtype])


def _dtype_from_bytes(buf: BufferLike) -> Tuple[int, Dtype]:
    if len(buf) < 4:
        raise ParseError
    dtype_len = int.from_bytes(buf[:4], byteorder="big")
    if len(buf) < 4 + dtype_len:
        raise ParseError
    dtype = bytes(buf[4 : 4 + dtype_len]).decode()
    bytes_read = 4 + dtype_len
    return bytes_read, dtype


def _data_to_bytes(data: bytes) -> bytes:
    b_len = len(data).to_bytes(4, byteorder="big")
    return b"".join([b_len, data])


def _data_from_bytes(buf: BufferLike) -> Tuple[int, bytes]:
    if len(buf) < 4:
        raise ParseError
    data_len = int.from_bytes(buf[:4], byteorder="big")
    if len(buf) < 4 + data_len:
        raise ParseError
    data = bytes(buf[4 : 4 + data_len])
    bytes_read = 4 + data_len
    return bytes_read, data


def _tensor_data_from_bytes(shape: Shape, dtype: Dtype, data: bytes) -> Any:
    tensor_data: Any

    if dtype == "bytes":
        assert len(shape) == 1 and shape[0] is None
        tensor_data = data
    elif dtype == "str":
        assert len(shape) == 1 and shape[0] is None
        tensor_data = data.decode()
    elif dtype in _ARRAY_LIKE:
        tensor_data = np.frombuffer(data, dtype=dtype).reshape(shape)
    else:
        raise ValueError(f"Unrecognized dtype {dtype}.")

    return tensor_data


def _tensor_data_to_bytes(tensor: Tensor) -> bytes:
    dtype = tensor.dtype
    data = tensor.data
    if dtype == "bytes":
        assert isinstance(data, bytes)
        return data
    if dtype == "str":
        assert isinstance(data, str)
        return data.encode()
    if dtype in _ARRAY_LIKE:
        assert isinstance(data, np.ndarray)
        return data.tobytes()
    raise ValueError(f"Unrecognized dtype {dtype}.")


__all__ = [
    "TensorPacket",
]
