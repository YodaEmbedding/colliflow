import socket
from typing import List

import rx
from rx.subject import Subject

from colliflow.model import Model
from colliflow.network.connectors import ClientsideConnector
from colliflow.serialization.mux_reader import read_mux_packet, start_reader
from colliflow.serialization.mux_writer import start_writer

from .module import ForwardAsyncModule


class TcpServerSubgraph(ForwardAsyncModule):
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
        self._start_stream(in_sock=self.in_sock, out_sock=self.out_sock)

    def forward(self, *inputs: rx.Observable) -> rx.Observable:
        for obs, subject in zip(inputs, self.inputs):
            obs.subscribe(subject)
        return self.outputs[0]

    def _start_stream(self, in_sock, out_sock):
        num_output_streams = len(self.graph._outputs)
        write = out_sock.sendall
        read = lambda: read_mux_packet(in_sock)
        start_writer(self.inputs, write)
        self.outputs = start_reader(num_output_streams, read)

    def _connect_server(self):
        """Open connection to data stream input/output sockets."""
        connector = ClientsideConnector(self.graph, "localhost", 5678)
        self.in_sock, self.out_sock = connector.connect()


__all__ = [
    "TcpServerSubgraph",
]
