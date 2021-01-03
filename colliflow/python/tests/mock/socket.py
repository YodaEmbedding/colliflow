import socket
from queue import Empty, Queue
from time import sleep


class LoopbackMockSocket(socket.socket):
    def __init__(self, blocking: bool = True, wait_all: bool = False):
        self._is_blocking = blocking
        self._is_wait_all = wait_all
        self._recv_buffer = b""
        self._queue = Queue()

    def recv(self, num_bytes: int) -> bytes:
        buf = bytearray(num_bytes)
        view = memoryview(buf)
        self.recv_into(view, num_bytes)
        return bytes(buf)

    def recv_into(self, buf: memoryview, num_bytes: int = 0) -> int:
        if num_bytes < 0:
            raise ValueError

        if num_bytes > len(buf):
            raise ValueError

        if num_bytes == 0:
            num_bytes = len(buf)

        offset = self._recv_step(buf, num_bytes, self._recv_buffer)

        # Always obtain something if is blocking socket
        if self._is_blocking and offset == 0:
            data = self._queue.get()
            offset += self._recv_step(buf[offset:], num_bytes - offset, data)

        while offset < num_bytes:
            try:
                data = (
                    self._queue.get()
                    if self._is_blocking and self._is_wait_all
                    else self._queue.get_nowait()
                )
            except Empty:
                break

            offset += self._recv_step(buf[offset:], num_bytes - offset, data)

        return offset

    def _recv_step(self, buf: memoryview, num_bytes: int, data: bytes) -> int:
        bytes_read = min(num_bytes, len(data))
        buf[:bytes_read] = data[:bytes_read]
        self._recv_buffer = data[bytes_read:]
        return bytes_read

    def sendall(self, buf: bytes):
        if len(buf) == 0:
            return
        self._queue.put(buf)
