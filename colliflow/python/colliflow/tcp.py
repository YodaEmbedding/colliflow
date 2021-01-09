import json
import socket
from asyncio import IncompleteReadError
from dataclasses import dataclass
from typing import Any, List, Tuple

import numpy as np

from colliflow.tensors import Tensor, TensorInfo
from colliflow.typing import Dtype, Shape

array_like = [
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
class TcpStreamMessageHeader:
    stream_id: int
    length: int


class TcpSocketStreamReader:
    def __init__(self, sock: socket.socket):
        self._sock = sock
        self._recv_buffer = bytearray()

    def readexactly(self, num_bytes: int) -> bytes:
        buf = bytearray(num_bytes)
        pos = 0
        while pos < num_bytes:
            n = self._recv_into(memoryview(buf)[pos:])
            if n == 0:
                raise IncompleteReadError(bytes(buf[:pos]), num_bytes)
            pos += n
        return bytes(buf)

    def readline(self) -> bytes:
        return self.readuntil(b"\n")

    def readuntil(self, separator: bytes = b"\n") -> bytes:
        if len(separator) != 1:
            raise ValueError("Only separators of length 1 are supported.")

        chunk = bytearray(4096)
        start = 0
        buf = bytearray(len(self._recv_buffer))
        bytes_read = self._recv_into(memoryview(buf))
        assert bytes_read == len(buf)

        while True:
            idx = buf.find(separator, start)
            if idx != -1:
                break

            start = len(self._recv_buffer)
            bytes_read = self._recv_into(memoryview(chunk))
            buf += memoryview(chunk)[:bytes_read]

        result = bytes(buf[: idx + 1])
        self._recv_buffer = b"".join(
            (memoryview(buf)[idx + 1 :], self._recv_buffer)
        )
        return result

    def readint(self, num_bytes: int = 4) -> int:
        return int.from_bytes(self.readexactly(num_bytes), byteorder="big")

    def readjson(self) -> Any:
        return json.loads(self.readline().decode())

    def _recv_into(self, view: memoryview) -> int:
        bytes_read = min(len(view), len(self._recv_buffer))
        view[:bytes_read] = self._recv_buffer[:bytes_read]
        self._recv_buffer = self._recv_buffer[bytes_read:]
        if bytes_read == len(view):
            return bytes_read
        bytes_read += self._sock.recv_into(view[bytes_read:])
        return bytes_read


class TcpSocketStreamWriter:
    def __init__(self, sock: socket.socket):
        self._sock = sock

    def write(self, data: bytes):
        self._sock.sendall(data)

    def writeline(self, msg: bytes):
        self.write(msg + b"\n")

    def writeint(self, num: int, num_bytes: int = 4):
        self.write(num.to_bytes(num_bytes, byteorder="big"))

    def writejson(self, obj: Any):
        self.writeline(json.dumps(obj).encode())


def tensor_from_bytes(dtype: Dtype, shape: Shape, data: bytes):
    tensor_data: Any

    if dtype == "bytes":
        assert len(shape) == 1 and shape[0] is None
        tensor_data = data
    elif dtype == "str":
        assert len(shape) == 1 and shape[0] is None
        tensor_data = data.decode()
    elif dtype in array_like:
        tensor_data = np.frombuffer(data, dtype=dtype).reshape(shape)
    else:
        raise ValueError(f"Unrecognized dtype {dtype}.")

    return Tensor(
        dtype=dtype,
        shape=shape,
        data=tensor_data,
    )


def tensor_to_bytes(tensor: Tensor) -> bytes:
    dtype = tensor.dtype
    data = tensor.data
    if dtype == "bytes":
        assert isinstance(data, bytes)
        return data
    if dtype == "str":
        assert isinstance(data, str)
        return data.encode()
    if dtype in array_like:
        assert isinstance(data, np.ndarray)
        return data.tobytes()
    raise ValueError(f"Unrecognized dtype {dtype}.")


class TcpTensorInputStream:
    def __init__(self, reader: TcpSocketStreamReader, infos: List[TensorInfo]):
        self._reader = reader
        self._num_streams = len(infos)
        self._dtypes = [x.dtype for x in infos]
        self._shapes = [x.shape for x in infos]
        self._buffers: List[bytes] = [b"" for _ in range(self._num_streams)]
        self._lengths: List[int] = [-1 for _ in range(self._num_streams)]

    def read_tensor(self) -> Tuple[int, Tensor]:
        """Read stream and returns stream id and tensor data."""
        while True:
            k = self._read_message()

            if self._lengths[k] == -1:
                data = self._buffers[k][:4]
                self._buffers[k] = self._buffers[k][4:]
                self._lengths[k] = int.from_bytes(data, byteorder="big")

            if len(self._buffers[k]) >= self._lengths[k]:
                break

        length = self._lengths[k]
        data = self._buffers[k][:length]
        self._buffers[k] = self._buffers[k][length:]
        self._lengths[k] = -1
        return k, tensor_from_bytes(self._dtypes[k], self._shapes[k], data)

    def _read_message(self):
        header = self._read_message_header()
        data = self._reader.readexactly(header.length)
        self._buffers[header.stream_id] += bytes(data)
        return header.stream_id

    def _read_message_header(self):
        stream_id = self._reader.readint(num_bytes=1)
        length = self._reader.readint()
        return TcpStreamMessageHeader(stream_id=stream_id, length=length)


class TcpTensorOutputStream:
    def __init__(
        self,
        writer: TcpSocketStreamWriter,
        num_streams: int,
    ):
        self._num_streams = num_streams
        self._writer = writer
        self._buffers: List[bytes] = [b"" for _ in range(self._num_streams)]

    def write_tensor(self, stream_id: int, tensor: Tensor):
        """Write tensor to muxer buffers."""
        data = tensor_to_bytes(tensor)
        num_bytes = len(data).to_bytes(4, byteorder="big")
        self._buffers[stream_id] += num_bytes + data

    def send_message(self, stream_id: int, length: int):
        """Send message consisting of length bytes taken from buffer."""
        data = self._buffers[stream_id][:length]
        self._buffers[stream_id] = self._buffers[stream_id][length:]
        if len(data) == 0:
            return
        self._writer.writeint(stream_id, num_bytes=1)
        self._writer.writeint(len(data))
        self._writer.write(data)


__all__ = [
    "TcpStreamMessageHeader",
    "TcpSocketStreamReader",
    "TcpSocketStreamWriter",
    "TcpTensorInputStream",
    "TcpTensorOutputStream",
]
