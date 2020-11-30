# USE THIS DOCUMENT FOR EXPLAINING AND OVERVIEW

import asyncio
import json
import socket
from asyncio import StreamReader, StreamWriter
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import numpy as np
import rx
import rx.subject
from rx import operators as ops

Shape = Union[Tuple[None], Tuple[int, ...]]


@dataclass
class TensorInfo:
    dtype: str
    shape: Shape


Graph = "Graph"
Tensor = "Tensor"

# Tensor?
# SymbolicTensor?

# SerializableGraphNode
# StaticLinkable


class Module:
    # config()
    # dtype?  <-- uhhh don't think this is a good idea here
    # shape?
    pass


# TODO How are Modules linked into a graph via the Functional __call__ API?


class ForwardModule(Module):
    """Function from M tensors to N tensors. M, N > 0."""

    def forward(self, *inputs: Tensor) -> List[Tensor]:
        raise NotImplementedError


class InputModule(Module):
    """Just like ForwardModule, but with forward being passthrough...?"""

    def forward(self, *inputs: Tensor) -> List[Tensor]:
        return inputs


class ForwardAsyncModule(Module):
    """Function from M observables to N observables. M, N > 0."""

    def forward(self, *inputs: rx.Observable) -> List[rx.Observable]:
        raise NotImplementedError


class InputAsyncModule(Module):
    """Outputs observables."""

    def produce(self) -> List[rx.Observable]:
        raise NotImplementedError


class OutputAsyncModule(Module):
    """Accepts observables."""

    def consume(self, *inputs: rx.Observable):
        raise NotImplementedError


# "Composable" hierarchical modules?

# What about "InputModules" that run synchronously when you call a model?
# e.g. preds = model(frames)

# TcpServer(AsyncModule) runs on client, creates the following two on server:
# TcpReceiver(InputAsyncModule)
# TcpSender(OutputAsyncModule)


# The following is handled by demuxer:
# StreamID: "Stream of X observable"


@dataclass
class TcpStreamMessageHeader:
    stream_id: int
    length: int


# The following is handled stream-by-stream:
# Tensor: DataType, Shape, Payload


@dataclass
class TensorMessageHeader:
    num_bytes: int
    # dtype: str
    # shape: Tuple[int]


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

    # def readline(self):
    #     while True:


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


def tensor_from_bytes(dtype: str, shape: Shape, data: bytes):
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
        dtypes = [x.dtype for x in self._infos]
        shapes = [x.shape for x in self._infos]
        # self._socket.connect()
        stream_reader = TcpSocketStreamReader(self._socket)
        self._stream = TcpTensorInputStream(stream_reader, dtypes, shapes)
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

# What are we trying to achieve using "AsyncModule"?
# Perhaps AsyncModule.forward should be renamed to "forward_rx".
# And Module should define "forward", which can be converted to "forward_rx".

# Graph execution order:
#
# This is probably what makes sense:
#
# .forward_rx() to set up observable connections
# .start() to run any initialization code
# begin pushing frames into pipeline
