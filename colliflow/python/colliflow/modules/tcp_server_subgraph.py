import socket
from typing import List, Tuple

import rx
from rx.subject import Subject

from colliflow.model import Model
from colliflow.network.connectors import ClientsideConnector
from colliflow.serialization.mux_reader import read_mux_packet, start_reader
from colliflow.serialization.mux_writer import start_writer

from .module import ForwardAsyncModule


class StreamingServerSubgraph(ForwardAsyncModule):
    def __init__(self, graph: Model):
        super().__init__(
            shape=graph._outputs[0].shape,
            dtype=graph._outputs[0].dtype,
        )

        self.graph = graph
        self.in_sock: socket.socket
        self.out_sock: socket.socket

        num_input_streams = len(self.graph._inputs)
        self.inputs = [Subject() for _ in range(num_input_streams)]
        self.outputs: List[rx.Observable]

    def setup(self):
        self._connect_server()
        self._start_stream()

    def forward(self, *inputs: rx.Observable) -> rx.Observable:
        for obs, subject in zip(inputs, self.inputs):
            obs.subscribe(subject)
        return self.outputs[0]

    def _connect_server(self):
        raise NotImplementedError

    def _start_stream(self):
        num_output_streams = len(self.graph._outputs)
        write = self.out_sock.sendall
        read = lambda: read_mux_packet(self.in_sock)
        start_writer(self.inputs, write)
        self.outputs = start_reader(num_output_streams, read)


class TcpServerSubgraph(StreamingServerSubgraph):
    def __init__(self, addr: Tuple[str, int], graph: Model):
        super().__init__(graph=graph)
        self.addr = addr

    def _connect_server(self):
        host, port = self.addr
        connector = ClientsideConnector(self.graph, host, port)
        self.in_sock, self.out_sock = connector.connect()


__all__ = [
    "StreamingServerSubgraph",
    "TcpServerSubgraph",
]
