import asyncio
import json
import random
from asyncio import StreamReader, StreamWriter
from pprint import pprint
from time import sleep, time

import rx
from rx import operators as ops
from rx.scheduler import NewThreadScheduler

from colliflow import Input, Model, Module, SymbolicTensor, Tensor, TensorInfo
from colliflow.modules import *

from .shared_modules import *

IP = "127.0.0.1"
PORT = 5678

epoch = time()


def get_time():
    return time() - epoch


def simple_model():
    client_func = lambda x: Tensor(shape=(14, 14, 512), dtype="uint8")
    server_func = lambda x: Tensor(shape=(1000,), dtype="float32")

    inputs = [
        Input(
            shape=(224, 224, 3),
            dtype="uint8",
        )
    ]

    x = inputs[0]
    # x = Preprocessor()(x)
    x = ClientInferenceModel(
        func=client_func,
        shape=(14, 14, 512),
        dtype="uint8",
    )(x)
    x = Postencoder()(x)
    x = Predecoder(
        shape=(14, 14, 512),
        dtype="uint8",
    )(x)
    x = ServerInferenceModel(  #
        func=server_func,
        shape=(1000,),
        dtype="float32",
    )(x)

    outputs = [x]

    return Model(inputs=inputs, outputs=outputs)


def server_model():
    client_func = lambda x: Tensor(shape=(14, 14, 512), dtype="uint8")
    server_func = lambda x: Tensor(shape=(1000,), dtype="float32")

    inputs = [
        FakeInput(
            shape=(224, 224, 3),
            dtype="uint8",
        )
    ]

    x = inputs[0]
    # x = Preprocessor()(x)
    x = ClientInferenceModel(
        func=client_func,
        shape=(14, 14, 512),
        dtype="uint8",
    )(x)
    x = Postencoder()(x)
    x = Predecoder(
        shape=(14, 14, 512),
        dtype="uint8",
    )(x)
    x = ServerInferenceModel(  #
        func=server_func,
        shape=(1000,),
        dtype="float32",
    )(x)

    outputs = [x]

    return Model(inputs=inputs, outputs=outputs)


class RandomMerge(Module):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def inner_config(self):
        return {}

    def forward(self, left: Tensor, right: Tensor):
        return left if random.random() < 0.5 else right

    def set_props_hook(self, left: SymbolicTensor, right: SymbolicTensor):
        assert left.shape == right.shape
        assert left.dtype == right.dtype
        self._shape = left.shape
        self._dtype = left.dtype


def multi_branch_model():
    client_func = lambda x: Tensor(shape=(14, 14, 512), dtype="uint8")
    server_func = lambda x: Tensor(shape=(1000,), dtype="float32")

    inputs = [
        Input(
            shape=(224, 224, 3),
            dtype="uint8",
        )
    ]
    a = inputs[0]

    x = Postencoder()(a)
    x = Predecoder(shape=(224, 224, 3), dtype="uint8")(x)
    b = x
    c = RandomMerge()(a, b)
    c = RandomMerge()(c, c)

    # x = Preprocessor()(c)
    x = ClientInferenceModel(
        func=client_func,
        shape=(14, 14, 512),
        dtype="uint8",
    )(x)
    x = Postencoder()(x)
    x = Predecoder(shape=(14, 14, 512), dtype="uint8")(x)
    x = ServerInferenceModel(  #
        func=server_func,
        shape=(1000,),
        dtype="float32",
    )(x)

    # outputs = [x, a, b, c]
    outputs = [x, c]
    # outputs = [x]

    return Model(inputs=inputs, outputs=outputs)


def model_client_server():
    client_func = lambda _: Tensor(shape=(14, 14, 512), dtype="uint8")
    server_func = lambda _: Tensor(shape=(1000,), dtype="float32")

    inputs = [Input(shape=(224, 224, 3), dtype="uint8")]
    x = inputs[0]
    # x = Preprocessor()(x)
    x = ClientInferenceModel(
        func=client_func, shape=(14, 14, 512), dtype="uint8"
    )(x)
    x = Postencoder()(x)
    # x = TcpSender()(x)
    outputs = [x]
    model_client = Model(inputs=inputs, outputs=outputs)

    inputs = [Input(shape=(None,), dtype="uint8")]
    x = inputs[0]
    x = Predecoder(shape=(14, 14, 512), dtype="uint8")(x)
    x = ServerInferenceModel(  #
        func=server_func, shape=(1000,), dtype="float32"
    )(x)
    # x = TcpSender()(x)
    outputs = [x]
    model_server = Model(inputs=inputs, outputs=outputs)

    return model_client, model_server


def main():
    print("\nSIMPLE MODEL CONSTRUCTION TEST")
    model = simple_model()
    print(model)

    print("\nSIMPLE SYNCHRONOUS PREDICTION TEST")
    preds = model(Tensor(shape=(224, 224, 3), dtype="uint8"))
    print(preds)

    print("\nSERIALIZE TEST")
    model_config = model.serialize_dict()
    pprint(model_config)
    # pprint(json.loads(json.dumps(model_config)))

    print("\nDESERIALIZE TEST")
    model = model_from_config(model_config)
    preds = model(Tensor(shape=(224, 224, 3), dtype="uint8"))
    print(preds)

    print("\nMODEL CLIENT/SERVER TEST")
    model_client, model_server = model_client_server()
    print(model_client)
    print(model_server)

    # print("\nCLIENT/SERVER EXECUTOR TEST")
    # client_executor = Executor(model=model_client)
    # server_executor = Executor(model=model_server)


def main_rx():
    frames = rx.interval(1).pipe(
        ops.do_action(lambda x: print(f"\n{get_time():.1f}  Frame {x}\n")),
        ops.map(lambda x: Tensor((224, 224, 3), "uint8")),
        ops.publish(),
    )
    model = simple_model()
    print(model)

    # TODO type checks to ensure always plugging in  correct amount of (,)
    preds = model(Tensor((224, 224, 3), "uint8"))
    print(preds)

    observables = model.to_rx(frames)
    observable = observables[0]
    observable.subscribe(lambda x: print(f"{get_time():.1f}  Result"))

    frames.connect()
    sleep(10)


async def tcp_echo_client(message):
    reader: StreamReader
    writer: StreamWriter
    reader, writer = await asyncio.open_connection(IP, PORT)

    print(f"Send: {message!r}")
    writer.write(message.encode())
    await writer.drain()

    print("Close the connection")
    writer.close()
    await writer.wait_closed()


def main_tcp():
    model = server_model()
    model_config = model.serialize()
    print(json.dumps(model.serialize_dict(), indent=2))
    asyncio.run(tcp_echo_client(model_config))


def main_tcp_redesign():
    def create_server_graph():
        inputs = [ServerTcpInput(shape=(None,), dtype="bytes")]
        x = inputs[0]
        # x = ServerTcpOutput(num_streams=len(inputs), sock=None)(x)
        stream_infos = [TensorInfo(x.shape, x.dtype)]
        x = ServerTcpOutput(num_streams=len(inputs), sock=None)(x)
        outputs = [x]
        return Model(inputs=inputs, outputs=outputs)

    def create_client_graph():
        inputs = [Input(shape=(None,), dtype="bytes")]
        x = inputs[0]
        x = ClientTcpServerSubgraph(
            addr=("localhost", 5678),
            graph=create_server_graph(),
        )(x)
        outputs = [x]
        return Model(inputs=inputs, outputs=outputs)

    client_model = create_client_graph()
    frames = rx.from_iterable(["abc", "def", "ghi"])
    outputs = client_model.to_rx(frames)
    outputs[0].subscribe(print, subscribe_on=NewThreadScheduler())


if __name__ == "__main__":
    # main()
    # main_rx()
    # main_tcp()
    main_tcp_redesign()
