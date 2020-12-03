# USE THIS DOCUMENT FOR EXPLAINING AND OVERVIEW

import asyncio
import inspect
import json
import socket
from asyncio import StreamReader, StreamWriter
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, cast

import numpy as np
import rx
import rx.subject
from rx import operators as ops

from colliflow.modules import _zip_observables
from colliflow.tensors import Dtype, Shape, SymbolicTensor, Tensor


@dataclass
class TensorInfo:
    dtype: Dtype
    shape: Shape


JsonDict = Dict[str, Any]
Graph = "Graph"

# SerializableGraphNode
# StaticLinkable

# TODO copy over relevant code here... e.g. serialization, observable construction, ...


@dataclass
class Node:
    input_nodes: List["Node"] = []
    output_nodes: List["Node"] = []
    input_shapes: List[Shape] = []
    input_dtypes: List[Dtype] = []
    output_shapes: List[Shape] = []
    output_dtypes: List[Dtype] = []


# TODO serialization
# TODO "to_rx" (done by Model/Graph?) (needs to recognize "input" modules)


class Module(Node):
    name: Optional[str] = None
    name_to_module: Dict[str, Type["Module"]] = {}
    registered_modules: List[Type["Module"]] = []

    def __init__(self, shape: Shape, dtype: Dtype):
        self.output_shapes = [shape]
        self.output_dtypes = [dtype]
        self._is_used_in_static_graph: bool = False

    def __init_subclass__(cls, **kwargs):
        """Registers module classes (required for serializability).

        If the class is registered as an "__AbstractModule",
        then the name registration is deferred to concrete submodules.
        """
        if cls.name == "__AbstractModule":
            cls.name = None
            super().__init_subclass__(**kwargs)
            return
        if cls.name is None:
            cls.name = cls.__name__
        cls.registered_modules.append(cls)
        cls.name_to_module[cls.name] = cls
        super().__init_subclass__(**kwargs)

    def __call__(self, *inputs: SymbolicTensor) -> SymbolicTensor:
        """Connects module instances into a static graph."""
        return self._set_graph_inputs(*inputs)

    def __repr__(self) -> str:
        kwargs_str = ", ".join(
            f"{k}={v!r}" for k, v in self.inner_config().items()
        )
        return f"{self.name}({kwargs_str})"

    def inner_config(self) -> JsonDict:
        """Override this to serialize custom module parameters."""
        raise NotImplementedError

    def to_rx(self, *inputs: rx.Observable) -> rx.Observable:
        """Produces output observable from input observables.

        See abstract subclass (e.g. `ForwardModule`) docstring
        for further details.
        """
        raise NotImplementedError

    def config(self, node_lut: Dict[Node, int]) -> JsonDict:
        """Returns serializable JSON dictionary.

        JSON dictionary describes graph connections and parameters
        so that a deserializer can reconstruct the module.
        """
        return {
            "id": node_lut[self],
            "name": self.name,
            "inputs": [node_lut.get(x, None) for x in self.input_nodes],
            "outputs": [node_lut.get(x, None) for x in self.output_nodes],
            "tensor_inputs": [
                (shape, dtype) for shape, dtype in
                zip(self.input_shapes, self.input_dtypes)
            ],
            "config": self.inner_config(),
        }

    @classmethod
    def from_config(cls, node_config: JsonDict) -> "Module":
        """Constructs module using JSON config."""
        name = node_config["name"]
        module_config = node_config["config"]
        create_module = cls.name_to_module[name]
        return create_module(**module_config)

    def _check_num_inputs(
        self, n_input: int, check_nodes=False, check_signature=False
    ):
        n_nodes = len(self.input_nodes)
        if check_nodes and n_input != n_nodes:
            raise ValueError(
                f"Length mismatch: expected {n_nodes} inputs, "
                f"actual {n_input}."
            )

    def _set_graph_inputs(self, *inputs: SymbolicTensor) -> SymbolicTensor:
        if self._is_used_in_static_graph:
            raise RuntimeError(
                "Cannot reuse a module instance in a static graph."
            )

        self._check_num_inputs(len(inputs), check_signature=True)

        # Link static graph nodes together
        for tensor in inputs:
            parent = tensor.parent
            self.input_nodes.append(parent)
            self.input_dtypes.append(tensor.dtype)
            self.input_shapes.append(tensor.shape)
            if parent is None:
                continue
            parent.output_nodes.append(self)

        self._is_used_in_static_graph = True

        outputs = [
            SymbolicTensor(shape, dtype, self)
            for shape, dtype in zip(self.output_shapes, self.output_dtypes)
        ]
        return outputs[0]


class ForwardModule(Module):
    """Function from M tensors to N tensors. M, N > 0."""

    name = "__AbstractModule"

    def to_rx(self, *inputs: rx.Observable) -> rx.Observable:
        """Produces output observable from input observables.

        Zips together the input observables,
        puts the resulting observable on the correct scheduler,
        and then runs `forward`.
        The output observable is multicast so it can be reused
        without worrying about recomputation.
        """
        self._check_num_inputs(len(inputs), check_nodes=True)
        observable = _zip_observables(*inputs)
        observable = observable.pipe(
            self._forward_to_rx_op(),
            ops.publish(),
        )
        observable = cast(rx.core.ConnectableObservable, observable)
        observable.connect()
        return observable

    def forward(self, *inputs: Tensor) -> Tensor:
        """Override to define this module's core functionality."""
        raise NotImplementedError

    def _forward_to_rx_op(self) -> Callable[[rx.Observable], rx.Observable]:
        return ops.map(lambda xs: self.forward(*xs))

    def _check_num_inputs(
        self, n_input: int, check_nodes=False, check_signature=False
    ):
        super()._check_num_inputs(n_input, check_nodes=check_nodes)
        n_forward = self._num_forward_params()
        if check_signature and n_input != n_forward:
            raise ValueError(
                f"Length mismatch: expected {n_forward} inputs, "
                f"actual {n_input}."
            )

    def _num_forward_params(self):
        return len(inspect.signature(self.forward).parameters)


class ForwardAsyncModule(Module):
    """Function from M observables to N observables. M, N > 0."""

    name = "__AbstractModule"

    def forward(self, *inputs: rx.Observable) -> rx.Observable:
        """Override to define this module's core functionality.

        Should return observables
        after transforming the input observables in some manner.
        """
        raise NotImplementedError

    def to_rx(self, *inputs: rx.Observable) -> rx.Observable:
        return self.forward(*inputs)


class InputAsyncModule(Module):
    """Produces Rx observables."""

    name = "__AbstractModule"

    def produce(self) -> rx.Observable:
        """Override to define this module's core functionality.

        Should create new observables and return them.
        """
        raise NotImplementedError

    def to_rx(self, *inputs: rx.Observable) -> rx.Observable:
        self._check_num_inputs(len(inputs), check_nodes=True)
        return self.produce().pipe(ops.publish())


class OutputAsyncModule(Module):
    """Consumes Rx observables."""

    name = "__AbstractModule"

    def consume(self, *inputs: rx.Observable):
        """Override to define this module's core functionality.

        Should subscribe to the input observables.
        """
        raise NotImplementedError

    def to_rx(self, *inputs: rx.Observable) -> rx.Observable:
        self._check_num_inputs(len(inputs), check_nodes=True)
        self.consume(*inputs)
        # Create dummy observable to satisfy type signature
        return rx.from_iterable([])


class InputModule(ForwardModule):
    """Passes input tensor through without any further processing."""

    name = "Input"

    def __init__(self, shape: Shape, dtype: Dtype, **kwargs):
        super().__init__(shape, dtype, **kwargs)
        self.input_shapes = [shape]
        self.input_dtypes = [dtype]

    def forward(self, input_: Tensor) -> Tensor:
        return input_


def Input(shape: Tuple[int], dtype: str):  # pylint: disable=invalid-name
    """Creates an input module of given shape and dtype."""
    x = SymbolicTensor(shape, dtype)
    return InputModule(shape, dtype)(x)


@dataclass
class TcpStreamMessageHeader:
    stream_id: int
    length: int


class TcpSocketStreamReader:
    def __init__(self, sock: socket.socket):
        self.sock = sock

    def readexactly(self, num_bytes: int):
        buf = bytearray(num_bytes)
        pos = 0
        while pos < num_bytes:
            n = self.sock.recv_into(memoryview(buf)[pos:])
            if n == 0:
                raise EOFError
            pos += n
        return buf

    def readint(self, num_bytes: int = 4):
        return int.from_bytes(self.readexactly(num_bytes), byteorder="big")

    def readjson(self):
        length = self.readint(4)
        msg = self.readexactly(length)
        return json.loads(msg.decode())


class TcpSocketStreamWriter:
    def __init__(self, sock: socket.socket):
        self.sock = sock

    def write(self, data: bytes):
        self.sock.sendall(data)

    def writeint(self, num: int, num_bytes: int = 4):
        data = num.to_bytes(num_bytes, byteorder="big")
        self.write(data)


array_like = [
    "int8",
    "int16",
    "int32",
    "int64",
    "uint8",
    "uint16",
    "uint32",
    "uint64",
    "float16",
    "float32",
    "float64",
]


def tensor_from_bytes(dtype: Dtype, shape: Shape, data: bytes):
    if dtype == "bytes":
        assert len(shape) == 1 and shape[0] is None
        tensor_data = data
    elif dtype == "str":
        assert len(shape) == 1 and shape[0] is None
        tensor_data = data.decode()
    elif dtype in array_like:
        tensor_data = np.frombuffer(data, dtype=dtype).reshape(shape)
    else:
        raise ValueError(f"Unrecognized dtype {dtype}.")

    return Tensor(
        dtype=dtype,
        shape=shape,
        data=tensor_data,
    )


def tensor_to_bytes(tensor: Tensor) -> bytes:
    dtype = tensor.dtype
    data = tensor.data
    if dtype == "bytes":
        assert isinstance(data, bytes)
        return data
    if dtype == "str":
        assert isinstance(data, str)
        return data.encode()
    if dtype in array_like:
        assert isinstance(data, np.ndarray)
        return data.tobytes()
    raise ValueError(f"Unrecognized dtype {dtype}.")


class TcpTensorInputStream:
    def __init__(self, reader: TcpSocketStreamReader, infos: List[TensorInfo]):
        self._reader = reader
        self._num_streams = len(infos)
        self._dtypes = [x.dtype for x in infos]
        self._shapes = [x.shape for x in infos]
        self._buffers: List[bytes] = [b"" for _ in range(self._num_streams)]
        self._lengths: List[int] = [-1 for _ in range(self._num_streams)]

    def read_tensor(self) -> Tuple[int, Tensor]:
        """Read stream and returns stream id and tensor data."""
        while True:
            k = self._read_message()

            if self._lengths[k] == -1:
                data = self._buffers[k][:4]
                self._buffers[k] = self._buffers[k][4:]
                self._lengths[k] = int.from_bytes(data, byteorder="big")

            if len(self._buffers[k]) >= self._lengths[k]:
                break

        length = self._lengths[k]
        data = self._buffers[k][:length]
        self._buffers[k] = self._buffers[k][length:]
        self._lengths[k] = -1
        return k, tensor_from_bytes(self._dtypes[k], self._shapes[k], data)

    def _read_message(self):
        header = self._read_message_header()
        data = self._reader.readexactly(header.length)
        self._buffers[header.stream_id] += bytes(data)
        return header.stream_id

    def _read_message_header(self):
        stream_id = self._reader.readint(num_bytes=1)
        length = self._reader.readint()
        return TcpStreamMessageHeader(stream_id=stream_id, length=length)


class TcpTensorOutputStream:
    def __init__(
        self,
        writer: TcpSocketStreamWriter,
        num_streams: int,
    ):
        self._num_streams = num_streams
        self._writer = writer
        self._buffers: List[bytes] = [b"" for _ in range(self._num_streams)]

    def write_tensor(self, stream_id: int, tensor: Tensor):
        """Write tensor to muxer buffers."""
        data = tensor_to_bytes(tensor)
        num_bytes = len(data).to_bytes(4, byteorder="big")
        self._buffers[stream_id] += num_bytes + data

    def send_message(self, stream_id: int, length: int):
        """Send message consisting of length bytes taken from buffer."""
        data = self._buffers[stream_id][:length]
        self._buffers[stream_id] = self._buffers[stream_id][length:]
        self._writer.writeint(stream_id, num_bytes=1)
        self._writer.writeint(length)
        self._writer.write(data)


class TcpReceiver(InputAsyncModule):
    def __init__(self, stream_infos: List[TensorInfo], sock: socket.socket):
        # self._socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._socket = sock
        self._infos = stream_infos
        self._num_streams = len(self._infos)
        self._stream: Optional[TcpTensorInputStream] = None

    def start(self):
        # self._socket.connect()
        stream_reader = TcpSocketStreamReader(self._socket)
        self._stream = TcpTensorInputStream(stream_reader, self._infos)
        self._network_reader = rx.from_iterable(self._reader()).pipe(
            ops.share(),
        )

    def produce(self) -> List[rx.Observable]:
        return [
            self._network_reader.pipe(
                ops.filter(lambda x: x[0] == i),
                ops.map(lambda x: x[1]),
            )
            for i in range(self._num_streams)
        ]

    def _reader(self):
        while True:
            yield self._stream.read_tensor()


class TcpSender(OutputAsyncModule):
    # TODO ensure that socket is NOT added to the serialization params
    def __init__(self, num_streams: int, sock: socket.socket):
        # self._socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._socket = sock
        self._num_streams = num_streams
        self._stream: Optional[TcpTensorOutputStream] = None

    def start(self):
        # self._socket.connect()
        stream_writer = TcpSocketStreamWriter(self._socket)
        self._stream = TcpTensorOutputStream(stream_writer, self._num_streams)
        message_requests = rx.from_iterable(self._sender())
        message_requests.subscribe(self._send_message)

    def consume(self, *inputs: rx.Observable):
        indexed_inputs = [
            x.pipe(ops.map(lambda x, i=i: (i, x)))
            for i, x in enumerate(inputs)
        ]
        zipped = rx.zip(*indexed_inputs)
        zipped.subscribe(self._writer)

    def _writer(self, tensor_pair: Tuple[int, Tensor]):
        stream_id, tensor = tensor_pair
        self._stream.write_tensor(stream_id, tensor)

    def _sender(self):
        length = 4096
        while True:
            for stream_id in range(self._num_streams):
                yield stream_id, length
            # TODO block until a buffer is non-empty

    def _send_message(self, message: Tuple[int, int]):
        stream_id, length = message
        self._stream.send_message(stream_id, length)


class TcpServer(ForwardAsyncModule):
    def __init__(
        self, addr: Tuple[str, int], graph: Graph, sock: socket.socket
    ):
        self._addr = addr
        self._graph = graph
        self._comm_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._sender: Optional[TcpSender] = None
        self._receiver: Optional[TcpReceiver] = None

    def start(self):
        self._establish_conn()
        # TODO plug socket into sender/receiver... can probably do in forward()
        self._sender.start()
        self._receiver.start()

    # NOTE forward is only used to provide information to set up observables?
    def forward(self, *inputs: rx.Observable) -> List[rx.Observable]:
        num_inputs = len(inputs)
        # output_infos = []

        self._sender = TcpSender(num_inputs, self._comm_socket)
        self._receiver = TcpReceiver(output_infos, self._comm_socket)

        self._sender.consume(*inputs)
        outputs = self._receiver.produce()

        return outputs

    def _establish_conn(self):
        self._comm_socket.connect(self._addr)
        msg = self._graph.serialize().encode()
        reader = TcpSocketStreamReader(self._comm_socket)
        writer = TcpSocketStreamWriter(self._comm_socket)
        writer.write(msg)
        d = reader.readjson()
        if d["status"] != "ready":
            raise Exception("Server could not be initialized with subgraph.")
        host = self._addr[0]
        port = d["port"]
        self._socket.connect((host, port))


class Server:
    def __init__(self, host: str = "localhost", port: int = 0):
        self._host = host
        self._port = port

    def start(self):
        asyncio.run(self.start_async())

    async def start_async(self):
        server = await asyncio.start_server(
            self.client_handler, self.host, self.port
        )
        await server.serve_forever()

    async def client_handler(self, reader: StreamReader, writer: StreamWriter):
        print("New client...")
        ip, port = writer.get_extra_info("peername")
        print(f"Connected to {ip}:{port}")

        line = await reader.readline()
        model = model_from_config(line.decode())
        print(model)

        # separate to_rx and start?
        # do we want to construct graph before start()?
        observables = model.to_rx([])

        observable = observables[0]
        observable.subscribe()  # on which thread?
        print("subscribed to first observable")

        # which sock? where is this used?
        # specify UDP in, TCP out? modules should handle it
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.bind((self._host, 0))
        port: int = sock.getsockname()[1]

        # isn't this blocking?
        sock.listen(1)

        response_dict = {"status": "ready", "port": port}
        response = json.dumps(response_dict) + "\n"
        writer.write(response.encode())
        await writer.drain()

        # start()/accept() on DIFFERENT thread to avoid blocking
        # only begin to recv after sock.accept
        conn, addr = sock.accept()

        model.set_sock(sock)
        observables = model.start()


server = Server(host="0.0.0.0", port=5678)


# TODO rewrite Model.to_rx
# TODO when sending subgraph, auto-surround with TcpReceiver/TcpSender?
# TODO plug sock

# TODO put various subscriptions on the correct threads
# TODO manage threads and hot, cold observables
# TODO thread-safe (especially write) buffers
# TODO start() communication properly via socket exchange
# TODO "state" machine which streams data after computing upload BW/ping
# TODO rate controls,  CONTIGUOUS_INTERVAL = 50ms  (computed from upload BW)
# TODO bi-directional time/rate sync

# TODO props hook for defining module output_shapes/dtypes at runtime
# TODO module multi-output
# TODO "composable" hierarchical modules?
# What about "InputModules" that run synchronously when you call a model?
# e.g. preds = model(frames)

# STAGES
#
# 0. User defines custom module classes
# 1. User connects modules into a graph
# 2. User "runs" graph:
#   2.1 module.start() for each module
#     2.1.1. Sub-graph is serialized
#     2.1.2. Sub-graph is transmitted to server
#     2.1.3. Server "runs" sub-graph
#       2.1.3.1. module.start() for each sub-graph module
#       2.1.3.1. module.forward() for each sub-graph module, then subscribe
#       2.1.3.3. Server responds that sub-graph is ready
#   2.2 module.forward() for each module, then subscribe

# Graph execution order:
#
# This is probably what makes sense:
#
# .start() to run any initialization code
# .forward_rx() to set up observable connections
# begin pushing frames into pipeline
