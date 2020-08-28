from time import sleep, time
from typing import Tuple, cast

import rx
import rx.operators as ops

from colliflow import (
    InputModule,
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


def FakeInput(
    shape: Tuple[int], dtype: str, scheduler=None
):  # pylint: disable=invalid-name
    return FakeInputLayer(shape, dtype, scheduler=scheduler)()


class FakeInputLayer(InputModule):
    name = "FakeInput"

    def __init__(self, shape: Tuple[int], dtype: str, **kwargs):
        super().__init__(shape, dtype, **kwargs)

    def inner_config(self):
        return {"shape": self.shape, "dtype": self.dtype}

    def to_rx(self):
        frames = rx.interval(1).pipe(
            ops.do_action(lambda x: print(f"\n{get_time():.1f}  Frame {x}\n")),
            ops.map(lambda _: self.forward()),
            ops.publish(),
        )
        frames = cast(rx.core.ConnectableObservable, frames)
        frames.connect()
        return frames

    def forward(self):
        return Tensor((224, 224, 3), "uint8")


def model_from_config(model_config) -> Model:
    client_func = lambda _: Tensor(shape=(14, 14, 512), dtype="uint8")
    server_func = lambda _: Tensor(shape=(1000,), dtype="float32")

    model = (
        Model.deserialize(model_config)
        if isinstance(model_config, str)
        else Model.deserialize_dict(model_config)
    )

    x = next(x for x in model.modules if isinstance(x, ClientInferenceModel))
    x.func = client_func
    x = next(x for x in model.modules if isinstance(x, ServerInferenceModel))
    x.func = server_func

    return model
