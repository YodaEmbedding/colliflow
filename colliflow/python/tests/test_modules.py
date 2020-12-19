import rx
import rx.operators as ops

from colliflow.modules import *
from colliflow.tensors import Tensor


def test_forward_module():
    class MyForwardModule(ForwardModule):
        def forward(self, x: Tensor) -> Tensor:
            return Tensor((None,), "str", str(x.data))

    inputs = [Tensor((1,), "int", x) for x in [1, 2, 3]]
    expected = [Tensor((None,), "str", x) for x in ["1", "2", "3"]]
    results = []

    module = MyForwardModule((None,), "str")
    obs = module.to_rx(rx.from_iterable(inputs))
    obs.subscribe(lambda x: results.append(x))

    assert results == expected


def test_input_module():
    inputs = [Tensor((1,), "int", x) for x in [1, 2, 3]]
    expected = inputs
    results = []

    module = InputModule((1,), "int")
    obs = module.to_rx(rx.from_iterable(inputs))
    obs.subscribe(lambda x: results.append(x))

    assert results == expected


def test_input_async_module():
    class MyInputAsyncModule(InputAsyncModule):
        def produce(self):
            return rx.from_iterable(inputs)

    inputs = [b"water", b"earth", b"fire", b"air"]
    expected = inputs
    results = []

    module = MyInputAsyncModule((None,), "bytes")
    obs = module.to_rx()
    obs.subscribe(lambda x: results.append(x))

    assert results == expected


def test_forward_async_module():
    class MyForwardAsyncModule(ForwardAsyncModule):
        def forward(self, xs: rx.Observable) -> rx.Observable:
            return xs.pipe(ops.map(str))

    inputs = [1, 2, 3]
    expected = ["1", "2", "3"]
    results = []

    module = MyForwardAsyncModule((None,), "str")
    obs = module.to_rx(rx.from_iterable(inputs))
    obs.subscribe(lambda x: results.append(x))

    assert results == expected


def test_output_async_module():
    class MyOutputAsyncModule(OutputAsyncModule):
        def __init__(self):
            # TODO shape and dtype aren't relevant for OutputAsyncModule
            super().__init__(shape=(None,), dtype="bytes")

        def consume(self, xs: rx.Observable):
            obs = xs.pipe(ops.do_action(lambda x: results.append(x)))
            obs.subscribe()

    inputs = ["once", "upon", "a", "midnight", "dreary"]
    expected = inputs
    results = []

    module = MyOutputAsyncModule()
    module.to_rx(rx.from_iterable(inputs))

    assert results == expected


# TODO uhhhh shouldn't the rx async modules be passing around Tensors?


# TODO:
# test inheritance... just check the name
# test TcpSender and TcpReceiver individually?
# test full fleged graph RUNNING (not serialization)

# TODO for ser/des, test input_nodes, output_nodes, etc
# type conversion...? test incorrect types, too
# test multithread
