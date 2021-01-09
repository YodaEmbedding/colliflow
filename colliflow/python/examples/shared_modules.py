from time import sleep, time
from typing import Tuple

import rx
import rx.operators as ops

from colliflow import InputAsyncModule, Model, Module, SymbolicTensor, Tensor

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


class TcpClient(Module):
    def __init__(self, hostname=None, port=None, **kwargs):
        super().__init__((None,), "uint8", **kwargs)
        self.hostname = hostname
        self.port = port

    def inner_config(self):
        return {}

    def forward(self, tensor: Tensor):
        return tensor


def FakeInput(shape: Tuple[int], dtype: str):  # pylint: disable=invalid-name
    return FakeInputLayer(shape, dtype)()


class FakeInputLayer(InputAsyncModule):
    name = "FakeInput"

    def __init__(self, shape: Tuple[int], dtype: str, **kwargs):
        super().__init__(shape, dtype, **kwargs)

    def inner_config(self):
        return {"shape": self.shape, "dtype": self.dtype}

    def produce(self):
        return rx.interval(1).pipe(
            ops.do_action(lambda x: print(f"\n{get_time():.1f}  Frame {x}\n")),
            ops.map(lambda _: Tensor((224, 224, 3), "uint8")),
            ops.share(),
        )


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
