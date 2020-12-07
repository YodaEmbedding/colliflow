import socket
from typing import Optional, Tuple

import rx
from rx import operators as ops
from rx.scheduler import ThreadPoolScheduler

from colliflow.modules.module import OutputAsyncModule
from colliflow.tcp import TcpSocketStreamWriter, TcpTensorOutputStream
from colliflow.tensors import Tensor
from colliflow.typing import JsonDict


class TcpSender(OutputAsyncModule):
    name = "__AbstractModule"

    def __init__(self, num_streams: int, sock: socket.socket):
        self._sock = sock
        self._num_streams = num_streams
        self._stream: Optional[TcpTensorOutputStream] = None

    def consume(self, *inputs: rx.Observable):
        self._create_network_writer()
        indexed_inputs = [
            obs.pipe(ops.map(lambda x, i=i: (i, x)))
            for i, obs in enumerate(inputs)
        ]
        zipped = rx.zip(*indexed_inputs)
        zipped.subscribe(self._writer)

    def _create_network_writer(self):
        stream_writer = TcpSocketStreamWriter(self._sock)
        self._stream = TcpTensorOutputStream(stream_writer, self._num_streams)
        message_requests = rx.from_iterable(self._sender())
        message_requests.subscribe(self._send_message)

    def _writer(self, tensor_pair: Tuple[int, Tensor]):
        stream_id, tensor = tensor_pair
        self._stream.write_tensor(stream_id, tensor)

    def _sender(self):
        length = 4096
        while True:
            for stream_id in range(self._num_streams):
                yield stream_id, length
            # TODO block until a buffer is non-empty

    def _send_message(self, message: Tuple[int, int]):
        stream_id, length = message
        self._stream.send_message(stream_id, length)


class ClientTcpSender(TcpSender):
    name = "ClientTcpSender"


class ServerTcpSender(TcpSender):
    name = "ServerTcpSender"

    def setup(self) -> JsonDict:
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._sock.bind(("0.0.0.0", 0))
        port: int = self._sock.getsockname()[1]

        # NOTE very tiny race condition till observable start
        io_scheduler = ThreadPoolScheduler()
        rx.from_callable(self._get_conn).subscribe(scheduler=io_scheduler)

        return {"port": port}

    def _get_conn(self):
        self._sock.listen(1)
        conn, _ = self._sock.accept()
        self._sock = conn
        # TODO send a setup() result indicating success after connection, too


def ServerTcpOutput(
    num_streams: int, sock: socket.socket
) -> ServerTcpSender:  # pylint: disable=invalid-name
    return ServerTcpSender(num_streams=num_streams, sock=sock)


__all__ = [
    "TcpSender",
    "ClientTcpSender",
    "ServerTcpSender",
    "ServerTcpOutput",
]
