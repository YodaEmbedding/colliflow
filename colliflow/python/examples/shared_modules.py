from time import sleep

import rx
import rx.operators as ops

from colliflow import InputModule, Module, SymbolicTensor, Tensor


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


class FakeInput(InputModule):
    def __init__(self, shape, dtype, **kwargs):
        super().__init__(shape, dtype, **kwargs)

    def inner_config(self):
        return {}

    def to_rx(self):
        frames = rx.interval(1).pipe(
            # ops.do_action(lambda x: print(f"\n{get_time():.1f}  Frame {x}\n")),
            ops.map(lambda x: Tensor((224, 224, 3), "uint8")),
            ops.publish(),
        )
        return frames
