import socket
from typing import TYPE_CHECKING, Optional, Tuple

import rx

from colliflow.modules.module import ForwardAsyncModule
from colliflow.modules.tcp_receiver import ClientTcpReceiver, ServerTcpReceiver
from colliflow.modules.tcp_sender import ClientTcpSender, ServerTcpSender
from colliflow.tcp import TcpSocketStreamReader, TcpSocketStreamWriter
from colliflow.tensors import TensorInfo
from colliflow.typing import JsonDict

if TYPE_CHECKING:
    from colliflow.model import Model


class ClientTcpServerSubgraph(ForwardAsyncModule):
    name = "ClientTcpServerSubgraph"

    def __init__(self, addr: Tuple[str, int], graph: "Model"):
        # TODO a bit hacky
        # DEBUG
        input_shapes = list(graph.modules[0].output_shapes)
        input_dtypes = list(graph.modules[0].output_dtypes)
        output_shapes = list(graph.modules[-1].input_shapes)
        output_dtypes = list(graph.modules[-1].input_dtypes)
        super().__init__(shape=output_shapes[0], dtype=output_dtypes[0])
        self.input_shapes = input_shapes
        self.input_dtypes = input_dtypes
        self.output_shapes = output_shapes
        self.output_dtypes = output_dtypes

        self._addr = addr
        self._graph = graph
        self._sender: Optional[ClientTcpSender] = None
        self._receiver: Optional[ClientTcpReceiver] = None
        self._input_stream_infos = [
            TensorInfo(shape=shape, dtype=dtype)
            for shape, dtype in zip(self.input_shapes, self.input_dtypes)
        ]
        self._output_stream_infos = [
            TensorInfo(shape=shape, dtype=dtype)
            for shape, dtype in zip(self.output_shapes, self.output_dtypes)
        ]

    def inner_config(self):
        return {"addr": self._addr, "graph": self._graph.serialize()}

    def forward(self, *inputs: rx.Observable) -> rx.Observable:
        comm_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        comm_sock.connect(self._addr)
        comm_writer = TcpSocketStreamWriter(comm_sock)
        comm_reader = TcpSocketStreamReader(comm_sock)
        line = (self._graph.serialize() + "\n").encode()
        comm_writer.write(line)

        for _ in range(len(self._graph.modules)):
            response = comm_reader.readjsonfixed()
            print(response)
            self._handle_setup_response(response)

        comm_sock.close()

        assert self._sender is not None
        assert self._receiver is not None

        self._sender.to_rx(*inputs)
        outputs = self._receiver.to_rx()

        return outputs

    def _handle_setup_response(self, response: JsonDict):
        module_id = response["module_id"]
        module = self._graph.modules[module_id]

        # TODO somewhat ad hoc and not very robust
        if isinstance(module, (ServerTcpReceiver, ServerTcpSender)):
            # Connect to server at random port specified by server
            host, _ = self._addr
            port = response["result"]["port"]
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.connect((host, port))

            # Server's TcpReceiver connects to client's TcpSender,
            # and vice versa
            if isinstance(module, ServerTcpReceiver):
                self._sender = ClientTcpSender(
                    num_streams=len(self._input_stream_infos), sock=sock
                )
            else:
                self._receiver = ClientTcpReceiver(
                    stream_infos=self._output_stream_infos, sock=sock
                )


__all__ = [
    "ClientTcpServerSubgraph",
]
