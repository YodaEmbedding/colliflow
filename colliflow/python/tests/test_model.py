from multiprocessing import Process
from time import sleep
from typing import List

import numpy as np
import rx
from rx.subject import Subject

from colliflow.model import Model
from colliflow.modules import *
from colliflow.modules.tcp_server_subgraph import (
    StreamingServerSubgraph,
    TcpServerSubgraph,
)
from colliflow.serialization.mux_reader import read_mux_packet, start_reader
from colliflow.serialization.mux_writer import start_writer
from colliflow.server import Server
from colliflow.tensors import Tensor
from colliflow.typing import Dtype, Shape

from .mock.socket import LoopbackMockSocket

WAIT_COLLECT = 0.01
WAIT_COLLECT_LONG = 0.1
WAIT_CONNECTION = 0.01


def test_run_empty_graph():
    def create_graph():
        inputs = [Input((1,), "int")]
        outputs = inputs
        return Model(inputs=inputs, outputs=outputs)

    inputs = [Tensor((1,), "int", x) for x in [1, 2, 3]]
    expected = inputs
    results = []

    model = create_graph()
    model.setup_blocking()
    obs = model.to_rx(rx.from_iterable(inputs))
    obs[0].subscribe(lambda x: results.append(x))

    assert results == expected


def test_run_simple_graph():
    class Stringify(ForwardModule):
        def forward(self, x: Tensor) -> Tensor:
            return Tensor((1,), "str", str(x.data))

    def create_graph():
        inputs = [Input((1,), "int")]
        x = inputs[0]
        x = Stringify((1,), "str")(x)
        outputs = [x]
        return Model(inputs=inputs, outputs=outputs)

    inputs = [Tensor((1,), "int", x) for x in [1, 2, 3]]
    expected = [Tensor((1,), "str", x) for x in ["1", "2", "3"]]
    results = []

    model = create_graph()
    model.setup_blocking()
    obs = model.to_rx(rx.from_iterable(inputs))
    obs[0].subscribe(lambda x: results.append(x))

    assert results == expected


def test_serverclient_intraprocess_graph():
    class IntraprocessServer(ForwardAsyncModule):
        def __init__(self, graph: Model):
            super().__init__(
                shape=graph._outputs[0].shape,
                dtype=graph._outputs[0].dtype,
            )
            self.graph = graph

        def inner_config(self):
            return {}

        def setup(self):
            return self.graph.setup_blocking()

        def forward(self, *inputs: rx.Observable) -> rx.Observable:
            xss = self.graph.to_rx(*inputs)
            return xss[0]

    def create_client_graph():
        inputs = [Input((1,), "int")]
        x = inputs[0]
        x = IntraprocessServer(graph=create_server_graph())(x)
        outputs = [x]
        return Model(inputs=inputs, outputs=outputs)

    def create_server_graph():
        inputs = [Input((1,), "int")]
        outputs = inputs
        return Model(inputs=inputs, outputs=outputs)

    inputs = [Tensor((1,), "int", x) for x in [1, 2, 3]]
    expected = inputs
    results = []

    model = create_client_graph()
    model.setup_blocking()
    obs = model.to_rx(rx.from_iterable(inputs))
    obs[0].subscribe(lambda x: results.append(x))

    assert results == expected


def test_serverclient_intraprocess_streaming_loopback_graph():
    class IntraprocessStreamingLoopbackServer(ForwardAsyncModule):
        def __init__(self, shape: Shape, dtype: Dtype):
            super().__init__(shape=shape, dtype=dtype)
            num_input_streams = 1
            self.inputs = [Subject() for _ in range(num_input_streams)]
            self.outputs: List[rx.Observable]

        def setup(self):
            num_output_streams = 1
            sock = LoopbackMockSocket()
            write = sock.sendall
            read = lambda: read_mux_packet(sock)
            start_writer(self.inputs, write)
            self.outputs = start_reader(num_output_streams, read)

        def forward(self, *inputs: rx.Observable) -> rx.Observable:
            for obs, subject in zip(inputs, self.inputs):
                obs.subscribe(subject)
            return self.outputs[0]

    def create_client_graph():
        inputs = [Input((1,), "int32")]
        x = inputs[0]
        x = IntraprocessStreamingLoopbackServer((1,), "int32")(x)
        outputs = [x]
        return Model(inputs=inputs, outputs=outputs)

    tensor_of = lambda x: Tensor((1,), "int32", np.array([x], dtype=np.int32))
    inputs = [tensor_of(x) for x in [1, 2, 3]]
    expected = inputs
    expected = inputs
    results = []

    model = create_client_graph()
    model.setup_blocking()
    obs = model.to_rx(rx.from_iterable(inputs))
    obs[0].subscribe(lambda x: results.append(x))

    sleep(WAIT_COLLECT)

    assert results == expected


def test_serverclient_intraprocess_streaming_graph():
    class IntraprocessStreamingServer(StreamingServerSubgraph):
        def _connect_server(self):
            self.in_sock = LoopbackMockSocket()
            self.out_sock = LoopbackMockSocket()

            self.graph.setup_blocking()

            num_input_streams = len(self.graph._inputs)
            write = self.in_sock.sendall
            read = lambda: read_mux_packet(self.out_sock)

            inputs = start_reader(num_input_streams, read)
            outputs = self.graph.to_rx(*inputs)
            start_writer(outputs, write)

    class Square(ForwardModule):
        def forward(self, x: Tensor):
            return Tensor(x.shape, x.dtype, x.data ** 2)

    def create_client_graph():
        inputs = [Input((1,), "int32")]
        x = inputs[0]
        x = IntraprocessStreamingServer(graph=create_server_graph())(x)
        outputs = [x]
        return Model(inputs=inputs, outputs=outputs)

    def create_server_graph():
        inputs = [Input((1,), "int32")]
        outputs = [Square(shape=(1,), dtype="int32")(inputs[0])]
        return Model(inputs=inputs, outputs=outputs)

    tensor_of = lambda x: Tensor((1,), "int32", np.array([x], dtype=np.int32))
    inputs = [tensor_of(x) for x in [1, 2, 3]]
    expected = [tensor_of(x) for x in [1, 4, 9]]
    results = []

    model = create_client_graph()
    model.setup_blocking()
    obs = model.to_rx(rx.from_iterable(inputs))
    obs[0].subscribe(lambda x: results.append(x))

    sleep(WAIT_COLLECT)

    assert results == expected


def test_serverclient_interprocess_streaming_graph():
    class Square(ForwardModule):
        def forward(self, x: Tensor):
            return Tensor(x.shape, x.dtype, x.data ** 2)

        def inner_config(self):
            return {
                "shape": self.input_shapes[0],
                "dtype": self.input_dtypes[0],
            }

    def create_client_graph():
        inputs = [Input((1,), "int32")]
        x = inputs[0]
        x = TcpServerSubgraph(
            addr=("localhost", 5678),
            graph=create_server_graph(),
        )(x)
        outputs = [x]
        return Model(inputs=inputs, outputs=outputs)

    def create_server_graph():
        inputs = [Input((1,), "int32")]
        outputs = [Square(shape=(1,), dtype="int32")(inputs[0])]
        return Model(inputs=inputs, outputs=outputs)

    tensor_of = lambda x: Tensor((1,), "int32", np.array([x], dtype=np.int32))
    inputs = [tensor_of(x) for x in [1, 2, 3]]
    expected = [tensor_of(x) for x in [1, 4, 9]]
    results = []

    start_local_server()

    model = create_client_graph()
    model.setup_blocking()
    obs = model.to_rx(rx.from_iterable(inputs))
    obs[0].subscribe(lambda x: results.append(x))

    sleep(WAIT_COLLECT_LONG)

    assert results == expected


def start_local_server():
    def run_local_server():
        server = Server("localhost", 5678)
        server.start()

    process = Process(target=run_local_server, daemon=True)
    process.start()
    sleep(WAIT_CONNECTION)


# TODO def test_complex_graph
# TODO def test_multibranch_graph
