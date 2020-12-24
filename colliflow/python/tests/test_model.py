import asyncio
from time import sleep

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


#####


import socket
import threading
from dataclasses import dataclass
from io import BytesIO
from queue import Queue
from threading import Thread
from typing import Any, Callable, List, Optional, Tuple, Union, cast

import numpy as np
from rx.scheduler import NewThreadScheduler
from rx.subject import Subject

from colliflow.typing import Dtype, Shape

BufferLike = Union[bytes, bytearray, memoryview]

# TODO put these in the correct files (create if needed)

# stream_serializer.py
# stream.py
# tcp.py
# tcp/sender.py
# tcp/sender.py
# packets/formats/?.py
# network/...?

# graph_serialization.py
# data_serialization.py


def shape_to_bytes(shape: Shape) -> bytes:
    """Converts Shape to serialized byte format.

    None values are replaced with -1.
    """
    shape = tuple(x if x is not None else -1 for x in shape)
    b_len = len(shape).to_bytes(4, byteorder="big")
    b_shape = b"".join(
        x.to_bytes(4, byteorder="big", signed=True) for x in shape
    )
    return b"".join([b_len, b_shape])


def shape_from_bytes(buf: BufferLike) -> Tuple[int, Shape]:
    """Converts serialized byte format to Shape."""
    if len(buf) < 4:
        raise ValueError
    shape_len = int.from_bytes(buf[:4], byteorder="big")
    if len(buf) < 4 + shape_len:
        raise ValueError
    chunks = [buf[4 * i + 4 : 4 * i + 8] for i in range(shape_len)]
    xs = [int.from_bytes(x, byteorder="big") for x in chunks]
    shape = tuple(x if x != -1 else None for x in xs)
    bytes_read = 4 + 4 * shape_len
    return bytes_read, cast(Shape, shape)


def dtype_to_bytes(dtype: str) -> bytes:
    """Converts Dtype to serialized byte format."""
    b_dtype = dtype.encode()
    b_len = len(b_dtype).to_bytes(4, byteorder="big")
    return b"".join([b_len, b_dtype])


def dtype_from_bytes(buf: BufferLike) -> Tuple[int, Dtype]:
    """Converts serialized byte format to Dtype."""
    if len(buf) < 4:
        raise ValueError
    dtype_len = int.from_bytes(buf[:4], byteorder="big")
    if len(buf) < 4 + dtype_len:
        raise ValueError
    dtype = bytes(buf[4 : 4 + dtype_len]).decode()
    bytes_read = 4 + dtype_len
    return bytes_read, dtype


def data_to_bytes(data: bytes) -> bytes:
    b_len = len(data).to_bytes(4, byteorder="big")
    return b"".join([b_len, data])


def data_from_bytes(buf: BufferLike) -> Tuple[int, bytes]:
    if len(buf) < 4:
        raise ValueError
    data_len = int.from_bytes(buf[:4], byteorder="big")
    if len(buf) < 4 + data_len:
        raise ValueError
    data = bytes(buf[4 : 4 + data_len])
    bytes_read = 4 + data_len
    return bytes_read, data


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


def tensor_data_from_bytes(shape: Shape, dtype: Dtype, data: bytes) -> Tensor:
    tensor_data: Any

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


def tensor_data_to_bytes(tensor: Tensor) -> bytes:
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


@dataclass
class TensorPacket:
    shape: Shape
    dtype: str
    data: bytes

    def to_bytes(self) -> bytes:
        return b"".join(
            [
                shape_to_bytes(self.shape),
                dtype_to_bytes(self.dtype),
                data_to_bytes(self.data),
            ]
        )

    def to_tensor(self) -> Tensor:
        return Tensor(
            shape=self.shape,
            dtype=self.dtype,
            data=tensor_data_from_bytes(self.shape, self.dtype, self.data),
        )

    @staticmethod
    def from_bytes(buf: BufferLike) -> Tuple[int, Optional["TensorPacket"]]:
        view = memoryview(buf)
        num_bytes = 0

        try:
            n, shape = shape_from_bytes(view[num_bytes:])
            num_bytes += n

            n, dtype = dtype_from_bytes(view[num_bytes:])
            num_bytes += n

            n, data = data_from_bytes(view[num_bytes:])
            num_bytes += n
        except ValueError:
            return 0, None

        tensor_packet = TensorPacket(shape=shape, dtype=dtype, data=data)

        return num_bytes, tensor_packet

    @staticmethod
    def from_tensor(tensor: Tensor) -> "TensorPacket":
        return TensorPacket(
            shape=tensor.shape,
            dtype=tensor.dtype,
            data=tensor_data_to_bytes(tensor),
        )


@dataclass
class MuxPacket:
    stream_id: int
    payload_size: int
    payload: bytes

    def to_bytes(self):
        b_stream_id = self.stream_id.to_bytes(4, byteorder="big")
        b_payload_size = self.payload_size.to_bytes(4, byteorder="big")
        return b"".join([b_stream_id, b_payload_size, self.payload])


# class MuxWriter:
#     """Mixes multiple streams into a single serialized byte stream.
#
#     Bytes written to the streams are stored in buffers.
#     A number of bytes may be flushed from a stream buffer.
#     """
#
#     def __init__(self, num_streams: int, sender: Callable[[bytes], None]):
#         self._sender = sender
#         self._queues = [Queue() for _ in range(num_streams)]
#         self._events = Queue()
#         self._buffers = [b"" for _ in range(num_streams)]
#
#     def save(self, stream_id: int, buf: bytes):
#         """Writes data to buffers for later sending."""
#         self._queues[stream_id].put(buf)
#         self._events.put(stream_id)
#
#     def write(self, stream_id: int, num_bytes: int):
#         """Sends bytes from a stream buffer via sender."""
#         chunks = []
#
#         if len(self._buffers[stream_id]) != 0:
#             data = self._buffers[stream_id]
#             chunk = data[:num_bytes]
#             self._buffers[stream_id] = data[num_bytes:]
#             num_bytes -= len(chunk)
#             chunks.append(chunk)
#
#         while num_bytes != 0 and not self._queues[stream_id].empty():
#             data = self._queues[stream_id].get()
#             chunk = data[:num_bytes]
#             self._buffers[stream_id] = data[num_bytes:]
#             num_bytes -= len(chunk)
#             chunks.append(chunk)
#
#         payload = b"".join(chunks)
#         mux_packet = MuxPacket(
#             stream_id=stream_id, payload_size=len(payload), payload=payload
#         )
#
#         self._sender(mux_packet.to_bytes())


# class NetworkSender:
#     """Sends data on its own persistent thread.
#
#     Call `start_loop` once to start sender thread.
#     Call `send` to send data in a non-blocking manner.
#     """
#
#     def __init__(self, writer: Callable[[bytes], None], controller):
#         self._writer = writer
#         self._controller = controller
#         self._queue = Queue()
#
#     def send(self, buf: bytes):
#         """Non-blocking send."""
#         self._queue.put(buf)
#
#     def start_loop(self):
#         thread = Thread(target=self._loop, daemon=True)
#         thread.start()
#
#     def _loop(self):
#         # print(">>> START LOOP")
#         while True:
#             # Alternatively, without controller, it's just:
#             # buf = self._queue.get()
#             # self._writer(buf)
#
#             while not self._queue.empty():
#                 buf = self._queue.get()
#                 self._writer(buf)
#
#             # TODO block...
#
#             # Controller isn't part of class description...
#             # it does seem a bit weird naming-wise to have this here...
#             # but then where else would it go?
#             #
#             # Note: make sure this is thread-safe
#             # self._controller.send_next()


class BufferNotifier:
    def __init__(self):
        self._size = 0
        self._changed = Queue()
        self._lock = threading.Lock()

    def wait_until_nonempty(self):
        """Blocks a thread until there is data to write."""
        while True:
            with self._lock:
                if self._size != 0:
                    break
                self._changed.queue.clear()
            self._changed.get()

    def notify_changed(self, offset: int):
        """Notify the waiting thread that buffer size has changed."""
        with self._lock:
            self._size += offset
            if self._changed.empty():
                self._changed.put(None)

    @property
    def size(self):
        return self._size


# TODO Rename to Muxer? Since it doesn't really WRITE anything...
# Though we could write stuff by merely adding a _write hook.
class MuxWriter:
    """Mixes multiple streams into a single serialized byte stream.

    Bytes written to the streams are stored in buffers.
    A number of bytes may be flushed from a stream buffer.

    Thread safety is only guaranteed for
    at most one thread using next_packet and
    at most one thread using wait_until_data_available.
    """

    def __init__(self, num_streams: int):
        self._queues = [Queue() for _ in range(num_streams)]
        self._buffers = [b"" for _ in range(num_streams)]
        self._notifier = BufferNotifier()
        self._notifiers = [BufferNotifier() for _ in range(num_streams)]

    def next_packet(self, stream_id: int, num_bytes: int) -> MuxPacket:
        """Returns MuxPacket containing data from a stream."""
        chunks = []

        # Pull data from _buffers first
        if len(self._buffers[stream_id]) != 0:
            data = self._buffers[stream_id]
            chunk = data[:num_bytes]
            self._buffers[stream_id] = data[num_bytes:]
            num_bytes -= len(chunk)
            chunks.append(chunk)

        # Pull data from _queues second
        while num_bytes != 0 and not self._queues[stream_id].empty():
            data = self._queues[stream_id].get()
            chunk = data[:num_bytes]
            self._buffers[stream_id] = data[num_bytes:]
            num_bytes -= len(chunk)
            chunks.append(chunk)

        payload = b"".join(chunks)
        payload_size = len(payload)
        mux_packet = MuxPacket(
            stream_id=stream_id, payload_size=payload_size, payload=payload
        )

        self._notifier.notify_changed(offset=-payload_size)
        self._notifiers[stream_id].notify_changed(offset=-payload_size)

        return mux_packet

    def save(self, stream_id: int, buf: bytes):
        """Saves data from specified stream for later muxing."""
        self._queues[stream_id].put(buf)
        self._notifier.notify_changed(offset=len(buf))
        self._notifiers[stream_id].notify_changed(offset=len(buf))

    def wait_until_data_available(self, stream_id: Optional[int] = None):
        """Blocks a thread until there is data to write."""
        if stream_id is None:
            self._notifier.wait_until_nonempty()
        else:
            self._notifiers[stream_id].wait_until_nonempty()

    @property
    def buffer_sizes(self) -> List[int]:
        """Returns buffer sizes in number of bytes for each stream."""
        return [x.size for x in self._notifiers]


class MuxWriterController:
    def __init__(self, mux_writer: MuxWriter):
        self._mux_writer = mux_writer

    def next_packet(self) -> MuxPacket:
        """Returns next packet for sending."""
        self._mux_writer.wait_until_data_available()
        buffer_sizes = self._mux_writer.buffer_sizes
        print(buffer_sizes)
        stream_id, num_bytes = next(
            (i, x) for i, x in enumerate(buffer_sizes) if x != 0
        )
        return self._mux_writer.next_packet(stream_id, num_bytes)


def start_writer_thread(
    controller: MuxWriterController, write: Callable[[bytes], None]
):
    def write_loop():
        while True:
            packet = controller.next_packet()
            # sock.sendall(packet.to_bytes())
            write(packet.to_bytes())

    thread = Thread(target=write_loop, daemon=True)
    thread.start()


def rx_mux(*xss: rx.Observable) -> rx.Observable:
    def pair_index(i: int) -> Callable[[Any], Any]:
        def inner(x: Any) -> Tuple[int, Any]:
            return i, x

        return inner

    paired = [xs.pipe(ops.map(pair_index(i))) for i, xs in enumerate(xss)]
    return rx.from_iterable(paired).pipe(ops.merge_all())


def mux_write(writer: MuxWriter, *inputs: rx.Observable):
    def write(pair: Tuple[int, Tensor]):
        stream_id, tensor = pair
        tensor_packet = TensorPacket.from_tensor(tensor)
        buf = tensor_packet.to_bytes()
        writer.save(stream_id, buf)

    rx_mux(*inputs).subscribe(write)


def start_reader_thread(subject: Subject, read: Callable[[], Any]):
    def read_loop():
        while True:
            subject.on_next(read())

    thread = Thread(target=read_loop, daemon=True)
    thread.start()


# TODO What is the most performant way to read data?
# Reading large buffer chunks? (e.g. 4096 or perhaps even larger)
# Perhaps 1MB with a timeout of 1ms if 1MB not available?
# Should this just send data into MuxReader without analyzing data?
# Perhaps this could be our implementation for now


def readexactly(sock: socket.socket, num_bytes: int):
    buf = bytearray(num_bytes)
    pos = 0
    while pos < num_bytes:
        cr = sock.recv_into(memoryview(buf)[pos:])
        if cr == 0:
            raise EOFError
        pos += cr
    return buf


def readint(sock: socket.socket, num_bytes: int = 4) -> int:
    return int.from_bytes(readexactly(sock, num_bytes), byteorder="big")


def read_mux_packet(sock: socket.socket) -> MuxPacket:
    stream_id = readint(sock)
    payload_size = readint(sock)
    payload = bytes(readexactly(sock, payload_size))
    return MuxPacket(
        stream_id=stream_id, payload_size=payload_size, payload=payload
    )


class TensorStreamDeserializer:
    def __init__(self, subject: Subject):
        self._subject = subject
        self._buffer = bytearray(b"")

    def on_next(self, buf: bytes):
        self._buffer += buf
        self._flush_buffer()

    def _flush_buffer(self):
        view = memoryview(self._buffer)
        is_changed = False

        while True:
            num_bytes, tensor_packet = TensorPacket.from_bytes(view)
            if tensor_packet is None:
                if is_changed:
                    self._buffer = bytearray(view)
                return
            is_changed = True
            tensor = tensor_packet.to_tensor()
            self._subject.on_next(tensor)
            view = view[num_bytes:]

        # tensor_packets, num_bytes = TensorPacket.many_from_bytes(
        #     self._buffer
        # )
        # self._buffer = self._buffer[num_bytes:]
        # for tensor_packet in tensor_packets:
        #     self._subject.on_next(tensor_packet)

        # while len(self._buffer) != 0:
        #     tensor_packet, num_bytes = TensorPacket.from_bytes(self._buffer)
        #     if tensor_packet is None:
        #         return
        #     self._subject.on_next(tensor_packet)
        #     self._buffer = self._buffer[num_bytes:]


def tensor_stream_deserializer(in_stream: rx.Observable) -> rx.Observable:
    out_stream = Subject()
    deserializer = TensorStreamDeserializer(out_stream)
    in_stream.pipe(
        ops.do_action(deserializer.on_next),
    ).subscribe()
    return out_stream


def mux_read(
    mux_packets: rx.Observable, num_streams: int
) -> List[rx.Observable]:
    return [
        mux_packets.pipe(
            ops.observe_on(NewThreadScheduler()),
            ops.filter(lambda x, i=i: x.stream_id == i),
            ops.map(lambda x: x.payload),
            tensor_stream_deserializer,
        )
        for i in range(num_streams)
    ]


class LoopbackMockSocket(socket.socket):
    def __init__(self):
        self._buffer = b""

    # TODO test with MSG_WAITALL disabled?
    def recv(self, n: int) -> bytes:
        chunk = self._buffer[:n]
        self._buffer = self._buffer[n:]
        return chunk

    def recv_into(self, buf: memoryview, num_bytes: int = 0) -> int:
        if len(buf) == 0:
            return 0
        while num_bytes == 0:
            num_bytes = len(self._buffer)
            sleep(0.001)
        num_bytes = min(num_bytes, len(buf))
        buf[:num_bytes] = self._buffer[:num_bytes]
        self._buffer = self._buffer[num_bytes:]
        return num_bytes

    def sendall(self, buf: bytes):
        sleep(0.05)
        self._buffer += buf


def test_serverclient_intraprocess_streaming_graph():
    class IntraprocessStreamingServer(ForwardAsyncModule):
        def __init__(self, graph: Model):
            super().__init__(
                shape=graph._outputs[0].shape,
                dtype=graph._outputs[0].dtype,
            )
            self.graph = graph
            # self._network_sender: NetworkSender

        def setup(self):
            # self._network_sender = NetworkSender(writer, controller)
            # self._network_sender.start_loop()
            # return self.graph.setup_blocking()
            # return asyncio.run(self._setup())
            pass

        def forward(self, *inputs: rx.Observable) -> rx.Observable:
            # TODO replace with conn to pseudo-"server" which transforms inputs
            sock = LoopbackMockSocket()

            write = sock.sendall
            num_input_streams = len(inputs)
            mux_writer = MuxWriter(num_input_streams)
            controller = MuxWriterController(mux_writer)
            mux_write(mux_writer, *inputs)
            start_writer_thread(controller, write)

            read = lambda: read_mux_packet(sock)
            # TODO measure num_output_streams properly; len(TcpOutput.outputs)
            # num_output_streams = len(self.graph._outputs)
            num_output_streams = 1  # DEBUG
            mux_packets = Subject()
            start_reader_thread(mux_packets, read)
            outputs = mux_read(mux_packets, num_output_streams)

            return outputs[0]

            # TODO why can't we reuse inputs observable...? Probably .connect()
            # TODO DEBUG
            # return inputs[0]

        # async def _setup(self):
        #     results = {}
        #     async for module_id, result in self.graph.setup():
        #         results[module_id] = result
        #     return results

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
    from multiprocessing import Process

    # p = Process(target=...)
    raise NotImplementedError


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
