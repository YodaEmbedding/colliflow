import socket
from typing import Any, Callable, Tuple

import rx
from rx import operators as ops
from rx.scheduler import ThreadPoolScheduler
from rx.subject import ReplaySubject

from colliflow.tcp import SocketStreamWriter, TcpTensorOutputStream
from colliflow.tensors import Tensor
from colliflow.typing import JsonDict

from .module import OutputAsyncModule


class TcpSender(OutputAsyncModule):
    name = "__AbstractModule"

    def __init__(self, num_streams: int, sock: socket.socket):
        # TODO technically, this module doesn't really need a shape or dtype
        super().__init__(shape=(None,), dtype="bytes")

        self._sock = sock
        self._num_streams = num_streams
        self._stream: TcpTensorOutputStream

    def inner_config(self):
        return {"num_streams": self._num_streams, "sock": None}

    def consume(self, *inputs: rx.Observable):
        self._create_network_writer()
        _rx_mux(*inputs).subscribe(self._writer)

    def _create_network_writer(self):
        stream_writer = SocketStreamWriter(self._sock)
        self._stream = TcpTensorOutputStream(stream_writer, self._num_streams)
        message_requests = rx.from_iterable(self._sender())
        message_requests.subscribe(self._send_message)

    def _writer(self, tensor_pair: Tuple[int, Tensor]):
        """Writes tensor to stream buffer.

        The buffer will later be flushed over the network.
        """
        stream_id, tensor = tensor_pair
        print(f"write {stream_id} {tensor}")
        self._stream.write_tensor(stream_id, tensor)

    # TODO none of this is thread safe!

    def _sender(self):
        """Controls how much data to send per stream."""
        length = 4096
        while True:
            for stream_id in range(self._num_streams):
                yield stream_id, length
            # TODO block until a buffer is non-empty... needs feedback

    def _send_message(self, message: Tuple[int, int]):
        """Sends a chunk of data of given length from a stream."""
        stream_id, length = message
        self._stream.send_message(stream_id, length)


class ClientTcpSender(TcpSender):
    name = "ClientTcpSender"


class ServerTcpSender(TcpSender):
    name = "ServerTcpSender"

    _is_conn_established: ReplaySubject

    def setup(self) -> JsonDict:
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._sock.bind(("0.0.0.0", 0))
        port: int = self._sock.getsockname()[1]
        self._is_conn_established = ReplaySubject()

        # NOTE very tiny race condition till observable start
        io_scheduler = ThreadPoolScheduler()
        rx.from_callable(self._get_conn).subscribe(scheduler=io_scheduler)

        return {"port": port}

    def consume(self, *inputs: rx.Observable):
        self._wait_for_conn()
        super().consume(*inputs)

    def _get_conn(self):
        print(f"Listening for {self.name} connection...")
        self._sock.listen(1)
        conn, _ = self._sock.accept()
        self._sock = conn
        print(f"Established {self.name} connection")
        self._is_conn_established.on_next(None)
        # self._is_conn_established.on_completed()
        # TODO send a setup() result indicating success after connection, too

    def _wait_for_conn(self):
        print("waiting")
        self._is_conn_established.subscribe()
        print("waiting end")


def ServerTcpOutput(  # pylint: disable=invalid-name
    num_streams: int, sock: socket.socket
) -> ServerTcpSender:
    return ServerTcpSender(num_streams=num_streams, sock=sock)


def _rx_mux(*xss: rx.Observable) -> rx.Observable:
    """Combines observables into single observable of indexed tuples.

    ```
    A:   --- A1 -------- A2 -- A3 ----------->
    B:   -------- B1 ----------------- B3 --->
                    [ rx_mux ]
    out: --- A1 - B1 --- A2 -- A3 ---- B3 --->
    ```

    The output events are of type `tuple[int, AOut | BOut]`,
    where the first item represents the stream index (A = 0, B = 1),
    and the second item holds the data.
    """
    ops.map

    def pair_index(i: int) -> Callable[[Any], Any]:
        def inner(x: Any) -> Tuple[int, Any]:
            return i, x

        return inner

    paired = [xs.pipe(ops.map(pair_index(i))) for i, xs in enumerate(xss)]
    return rx.from_iterable(paired).pipe(ops.merge_all())


__all__ = [
    "TcpSender",
    "ClientTcpSender",
    "ServerTcpSender",
    "ServerTcpOutput",
]
