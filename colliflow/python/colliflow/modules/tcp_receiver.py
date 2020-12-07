import socket
from typing import List, Optional

import rx
from rx import operators as ops
from rx.scheduler import ThreadPoolScheduler

from colliflow.modules.module import InputAsyncModule
from colliflow.tcp import TcpSocketStreamReader, TcpTensorInputStream
from colliflow.tensors import SymbolicTensor, TensorInfo
from colliflow.typing import Dtype, JsonDict, Shape


class TcpReceiver(InputAsyncModule):
    name = "__AbstractModule"

    def __init__(self, stream_infos: List[TensorInfo], sock: socket.socket):
        self._sock = sock
        self._infos = stream_infos
        self._num_streams = len(self._infos)
        self._stream: Optional[TcpTensorInputStream] = None

    def produce(self) -> List[rx.Observable]:
        self._create_network_reader()
        return [
            self._network_reader.pipe(
                ops.filter(lambda x, i=i: x[0] == i),
                ops.map(lambda x: x[1]),
            )
            for i in range(self._num_streams)
        ]

    def _create_network_reader(self):
        stream_reader = TcpSocketStreamReader(self._sock)
        self._stream = TcpTensorInputStream(stream_reader, self._infos)
        self._network_reader = rx.from_iterable(self._reader()).pipe(
            ops.share(),
        )

    def _reader(self):
        while True:
            yield self._stream.read_tensor()


class ClientTcpReceiver(TcpReceiver):
    name = "ClientTcpReceiver"


class ServerTcpReceiver(TcpReceiver):
    name = "ServerTcpReceiver"

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


def ServerTcpInput(
    shape: Shape, dtype: Dtype, sock: Optional[socket.socket] = None
) -> SymbolicTensor:  # pylint: disable=invalid-name
    info = TensorInfo(shape=shape, dtype=dtype)
    module = ServerTcpReceiver([info], sock=sock)
    x = SymbolicTensor(shape=shape, dtype=dtype, parent=module)
    x.parent = module
    return x


__all__ = [
    "TcpReceiver",
    "ClientTcpReceiver",
    "ServerTcpReceiver",
    "ServerTcpInput",
]
