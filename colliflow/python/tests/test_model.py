from time import sleep
from typing import Any, Callable, List, Sequence

import numpy as np
import rx
from rx.subject import Subject

from colliflow.model import Model
from colliflow.modules import *
from colliflow.serialization.mux_reader import (
    mux_read,
    read_mux_packet,
    start_reader_thread,
)
from colliflow.serialization.mux_writer import (
    MuxWriter,
    MuxWriterController,
    mux_write,
    start_writer_thread,
)
from colliflow.tensors import Tensor

from .mock.socket import LoopbackMockSocket


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
    class MyForwardModule(ForwardModule):
        def __init__(self):
            super().__init__((1,), "str")

        def forward(self, x: Tensor) -> Tensor:
            # TODO perhaps insert an auto-assert on tensor's type and shape
            return Tensor(self.shape, self.dtype, str(x.data))

    def create_graph():
        inputs = [Input((1,), "int")]
        x = inputs[0]
        x = MyForwardModule()(x)
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
        def forward(self, *inputs: rx.Observable) -> rx.Observable:
            sock = LoopbackMockSocket()
            write = sock.sendall
            read = lambda: read_mux_packet(sock)
            start_writer(inputs, write)
            outputs = start_reader(len(inputs), read)
            return outputs[0]

    def create_client_graph():
        inputs = [Input((1,), "int32")]
        x = inputs[0]
        x = IntraprocessStreamingLoopbackServer((1,), "int32")(x)
        outputs = [x]
        return Model(inputs=inputs, outputs=outputs)

    inputs = [
        Tensor((1,), "int32", np.array([x], dtype=np.int32)) for x in [1, 2, 3]
    ]
    expected = inputs
    results = []

    model = create_client_graph()
    model.setup_blocking()
    obs = model.to_rx(rx.from_iterable(inputs))
    obs[0].subscribe(lambda x: results.append(x))
    sleep(0.3)

    assert results == expected


def test_serverclient_intraprocess_streaming_graph():
    class IntraprocessStreamingServer(ForwardAsyncModule):
        def __init__(self, graph: Model):
            super().__init__(
                shape=graph._outputs[0].shape,
                dtype=graph._outputs[0].dtype,
            )
            self.graph = graph
            self.in_sock = LoopbackMockSocket()
            self.out_sock = LoopbackMockSocket()

        def setup(self):
            self._start_server(in_sock=self.out_sock, out_sock=self.in_sock)

        def forward(self, *inputs: rx.Observable) -> rx.Observable:
            # TODO measure num_output_streams properly; len(TcpOutput.outputs)
            # num_output_streams = len(self.graph._outputs)
            num_output_streams = 1  # DEBUG
            write = self.out_sock.sendall
            read = lambda: read_mux_packet(self.in_sock)

            start_writer(inputs, write)
            outputs = start_reader(num_output_streams, read)

            return outputs[0]

        def _start_server(self, in_sock, out_sock):
            self.graph.setup_blocking()

            num_input_streams = 1  # DEBUG
            write = out_sock.sendall
            read = lambda: read_mux_packet(in_sock)

            inputs = start_reader(num_input_streams, read)
            outputs = self.graph.to_rx(*inputs)
            start_writer(outputs, write)

            # TODO perhaps no output generated due to ConnectableObservable (?)
            # bug that we were wondering about

    def create_client_graph():
        inputs = [Input((1,), "int32")]
        x = inputs[0]
        x = IntraprocessStreamingServer(graph=create_server_graph())(x)
        outputs = [x]
        return Model(inputs=inputs, outputs=outputs)

    def create_server_graph():
        inputs = [Input((1,), "int32")]
        outputs = inputs
        return Model(inputs=inputs, outputs=outputs)

    inputs = [
        Tensor((1,), "int32", np.array([x], dtype=np.int32)) for x in [1, 2, 3]
    ]
    expected = inputs
    results = []

    model = create_client_graph()
    model.setup_blocking()
    obs = model.to_rx(rx.from_iterable(inputs))
    obs[0].subscribe(lambda x: results.append(x))

    sleep(0.3)

    assert results == expected


# TODO replace loopback socket with actual graph


def test_serverclient_graph():
    # TODO automatically open new python process for "server"
    # from multiprocessing import Process
    # p = Process(target=...)
    raise NotImplementedError


def start_writer(
    inputs: Sequence[rx.Observable], write: Callable[[bytes], None]
):
    num_input_streams = len(inputs)
    mux_writer = MuxWriter(num_input_streams)
    controller = MuxWriterController(mux_writer)
    mux_write(mux_writer, *inputs)
    start_writer_thread(controller, write)


def start_reader(
    num_streams: int, read: Callable[[], Any]
) -> List[rx.Observable]:
    mux_packets = Subject()
    start_reader_thread(mux_packets, read)
    return mux_read(mux_packets, num_streams)


# def test_complex_graph
# def test_multibranch_graph
# def test_serverclient_graph

# MAIN COMPONENTS:
#
# - Data stream mux/demux serialization
# - Data stream controller (determines how much data to send per stream)
# - Data tcp stream
# - Tcp stream setup
#
# Separate these components into composable classes

# test "fake communication" or "fake subgraph"?
# Test TcpSubgraph style module but without actual Tcp?

# Test incrementally...
# e.g. test the DATA serialization functionality
# then test the DATA "read/write" on actual threads
# then test with proper TCP
# then test with "Server()-based subgraph runner"

# If you test incrementally, you can have following benefits:
#
#   - "guarantee" of stuff working
#   - ability to decouple code into separate testable units
#   - reduced TcpSender/TcpReceiver/etc complexity
