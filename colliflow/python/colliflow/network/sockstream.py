import json
import socket
from asyncio import IncompleteReadError
from typing import Any

WAIT_CONNECTION = 0.01


class SocketStreamReader:
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


class SocketStreamWriter:
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
