import random
from pprint import pprint
from time import sleep, time

import rx
from rx import operators as ops

from colliflow import (
    Input,
    Model,
    Module,
    SymbolicTensor,
    Tensor,
)

epoch = time()


def get_time():
    return time() - epoch


class Preprocessor(Module):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def inner_config(self):
        return {}

    def forward(self, tensor: Tensor):
        return tensor

    def set_props_hook(self, tensor: SymbolicTensor):
        self._shape = tensor.shape
        self._dtype = tensor.dtype


class ClientInferenceModel(Module):
    def __init__(self, func=None, shape=None, dtype=None, **kwargs):
        super().__init__(shape, dtype, **kwargs)
        self.func = func

    def inner_config(self):
        return {"shape": self.shape, "dtype": self.dtype}

    def forward(self, tensor: Tensor):
        sleep(0.7)
        return self.func(tensor)


class ServerInferenceModel(Module):
    def __init__(self, func=None, shape=None, dtype=None, **kwargs):
        super().__init__(shape, dtype, **kwargs)
        self.func = func

    def inner_config(self):
        return {"shape": self.shape, "dtype": self.dtype}

    def forward(self, tensor: Tensor):
        sleep(0.5)
        return self.func(tensor)


class Postencoder(Module):
    def __init__(self, **kwargs):
        super().__init__((None,), "uint8", **kwargs)

    def inner_config(self):
        return {}

    def forward(self, tensor: Tensor):
        return tensor


class Predecoder(Module):
    def __init__(self, shape, dtype, **kwargs):
        super().__init__(shape, dtype, **kwargs)

    def inner_config(self):
        return {"shape": self.shape, "dtype": self.dtype}

    def forward(self, tensor: Tensor):
        return tensor


# Not really needed since Postencoder already converts to uint8
# class TcpSender(Module):
#     def __init__(self):
#         super().__init__((None,), "uint8")
#
#     def inner_config(self):
#         return {}
#
#     def forward(self, tensor: Tensor):
#         return Tensor(self.shape, self.dtype)


class TcpClient(Module):
    def __init__(self, hostname=None, port=None, **kwargs):
        super().__init__((None,), "uint8", **kwargs)
        self.hostname = hostname
        self.port = port

    def inner_config(self):
        return {}

    def forward(self, tensor: Tensor):
        return tensor


# class TcpServer(Module):
#     # so... this should what? keep a bind listener open? then wait for input?
#     # input should be in a "streaming" tensor format...! (byte header/etc)
#
#     def __init__(self, hostname=None, port=None, **kwargs):
#         super().__init__((None,), "uint8", **kwargs)
#         self.hostname = hostname
#         self.port = port
#         # wait... what about "async" server that we wrote? nevermind that?
#         # that's useful for multiclient architecture... but forget that for now
#
#         # perhaps pass a prepared socket in? is that what the executor's job is?
#         # Meh... actually, it's fine to do this I think but not IMMEDIATELY?
#         # only when model is "initialized" properly, and we send a start signal
#         # to all modules for any initialization code?
#
#
#         # does this module really need an "input"? or can it be a "producer"
#         # node (e.g. like InputLayer)
#
#         # also, this is all running in a separate thread...
#         # OH SO THATS WHAT EXECUTOR DOES! MANAGE THREADS! and pools!
#
#         # Design all these things on Surface...
#
#         self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
#         self.sock.bind((hostname, port))
#         self.sock.listen(1)
#         # self.conn, self.addr = self.sock.accept()
#         # self.conn.recv()
#
#         # TODO
#
#     def inner_config(self):
#         return {}
#
#     def forward(self, tensor: Tensor):
#         return tensor


def simple_model():
    client_func = lambda x: Tensor(shape=(14, 14, 512), dtype="uint8")
    server_func = lambda x: Tensor(shape=(1000,), dtype="float32")

    inputs = [Input(shape=(224, 224, 3), dtype="uint8", scheduler="io")]

    x = inputs[0]
    x = Preprocessor(scheduler="cpu")(x)
    x = ClientInferenceModel(
        func=client_func, shape=(14, 14, 512), dtype="uint8", scheduler="cpu"
    )(x)
    x = Postencoder(scheduler="cpu")(x)
    x = Predecoder(shape=(14, 14, 512), dtype="uint8", scheduler="cpu")(x)
    x = ServerInferenceModel(  #
        func=server_func, shape=(1000,), dtype="float32", scheduler="cpu"
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

    inputs = [Input(shape=(224, 224, 3), dtype="uint8", scheduler="io")]
    a = inputs[0]

    x = Postencoder(scheduler="cpu")(a)
    x = Predecoder(shape=(224, 224, 3), dtype="uint8")(x)
    b = x
    c = RandomMerge()(a, b)
    c = RandomMerge()(c, c)

    x = Preprocessor(scheduler="cpu")(c)
    x = ClientInferenceModel(
        func=client_func, shape=(14, 14, 512), dtype="uint8", scheduler="cpu"
    )(x)
    x = Postencoder()(x)
    x = Predecoder(shape=(14, 14, 512), dtype="uint8")(x)
    x = ServerInferenceModel(  #
        func=server_func, shape=(1000,), dtype="float32", scheduler="cpu"
    )(x)

    # outputs = [x, a, b, c]
    outputs = [x, c]
    # outputs = [x]

    return Model(inputs=inputs, outputs=outputs)


def model_client_server():
    client_func = lambda x: Tensor(shape=(14, 14, 512), dtype="uint8")
    server_func = lambda x: Tensor(shape=(1000,), dtype="float32")

    inputs = [Input(shape=(224, 224, 3), dtype="uint8")]
    x = inputs[0]
    x = Preprocessor()(x)
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


def model_from_config(model_config):
    client_func = lambda x: Tensor(shape=(14, 14, 512), dtype="uint8")
    server_func = lambda x: Tensor(shape=(1000,), dtype="float32")

    model = Model.deserialize_dict(model_config)
    x = next(x for x in model.modules if isinstance(x, ClientInferenceModel))
    x.func = client_func
    x = next(x for x in model.modules if isinstance(x, ServerInferenceModel))
    x.func = server_func

    return model


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

    # preds = model(Tensor((224, 224, 3), "uint8"))
    # print(preds)

    observables = model.to_rx(frames)
    observable = observables[0]
    observable.subscribe(lambda x: print(f"{get_time():.1f}  Result"))

    frames.connect()
    sleep(10)

    # TODO custom Request(1) graph?
    # When should "output" TCP node send Request(1) notification?
    # When it (thinks) that the receiver has received all data.

    # "Publish" makes everything a hot observable.
    # I assume that when .onNext() is called on a Subject, it multicasts to all
    # observers "immediately". If the observer is not ready, it probably pushes
    # it into some buffer of the subscriber.

    # TODO
    # IndexedTensor
    # buffer_size, drop=True, etc
    # backpressure
    # time slice scheduling
    # scheduler messaging
    # tcp messaging
    # tcp client/server
    # ordering: "tensor indexes" (e.g. tcp "enforces" order; udp order lost tensors)
    # ...

    # NOTE
    # predictive early-dropping frames + scheduling matters for low-latency
    # ...not really for high-throughput only, though

    # TEST
    # multicasting
    # various architectures, fuzzy timings/delays
    # verify computation completes, no deadlocks, ...


if __name__ == "__main__":
    # main()
    main_rx()


# TODO flow of actual data...? ehhhh

# TODO
# Test entire model (with local TCP)

# TODO
# Communication protocol... does TcpSender handle it? Or does the actual
# executor/context determine what to do? What's the purpose of TcpSender, then?
#
# Doesn't it make the API cleaner to let TcpSender handle certain details?
#
# But it also makes sense to have the "Executor" switch between models
# (receiving/sending configs), sending stats/etc, and so on.
# Perhaps TcpSender is then just some sort of interface that outputs "packets"
# (or in this case, a data stream) that the graph executor itself then relays.

# TODO
# Should we have each "Module" node report estimated compute times?
# That way, we can schedule better?
