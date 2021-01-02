import socket
from time import sleep


class LoopbackMockSocket(socket.socket):
    def __init__(self):
        self._buffer = b""

    def recv(self, n: int) -> bytes:
        chunk = self._buffer[:n]
        self._buffer = self._buffer[n:]
        return chunk

    def recv_into(self, buf: memoryview, num_bytes: int = 0) -> int:
        if len(buf) == 0:
            return 0
        while num_bytes == 0:
            num_bytes = len(self._buffer)
            sleep(0.001)
        num_bytes = min(num_bytes, len(buf))
        buf[:num_bytes] = self._buffer[:num_bytes]
        self._buffer = self._buffer[num_bytes:]
        return num_bytes

    def sendall(self, buf: bytes):
        sleep(0.05)
        self._buffer += buf
