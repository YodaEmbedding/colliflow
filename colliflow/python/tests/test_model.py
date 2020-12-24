import asyncio

import rx
import rx.operators as ops

from colliflow.model import Model
from colliflow.modules import *
from colliflow.tensors import Tensor


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

        def setup(self):
            return asyncio.run(self._setup())

        def forward(self, *inputs: rx.Observable) -> rx.Observable:
            xss = self.graph.to_rx(*inputs)
            return xss[0]

        async def _setup(self):
            results = {}
            async for module_id, result in self.graph.setup():
                results[module_id] = result
            return results

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


def test_serverclient_intraprocess_streaming_graph():
    class IntraprocessStreamingServer(ForwardAsyncModule):
        def __init__(self, graph: Model):
            super().__init__(
                shape=graph._outputs[0].shape,
                dtype=graph._outputs[0].dtype,
            )
            self.graph = graph

        def setup(self):
            return asyncio.run(self._setup())

        def forward(self, *inputs: rx.Observable) -> rx.Observable:
            # TODO ser/des for DATA
            raise NotImplementedError

        async def _setup(self):
            results = {}
            async for module_id, result in self.graph.setup():
                results[module_id] = result
            return results

    def create_client_graph():
        inputs = [Input((1,), "int")]
        x = inputs[0]
        x = IntraprocessStreamingServer(graph=create_server_graph())(x)
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


def test_serverclient_graph():
    # TODO automatically open new python process for "server"
    from multiprocessing import Process

    # p = Process(target=...)
    raise NotImplementedError


# def test_complex_graph
# def test_multibranch_graph
# def test_serverclient_graph


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
