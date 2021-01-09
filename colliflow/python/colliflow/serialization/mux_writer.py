import threading
from queue import Queue
from threading import Thread
from typing import Any, Callable, List, Optional, Sequence, Tuple

import rx
import rx.operators as ops

from colliflow.tensors import Tensor

from .mux_packet import MuxPacket
from .tensor_packet import TensorPacket


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
        stream_id, num_bytes = next(
            (i, x) for i, x in enumerate(buffer_sizes) if x != 0
        )
        return self._mux_writer.next_packet(stream_id, num_bytes)


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


def start_writer(
    inputs: Sequence[rx.Observable], write: Callable[[bytes], None]
):
    num_input_streams = len(inputs)
    mux_writer = MuxWriter(num_input_streams)
    controller = MuxWriterController(mux_writer)
    mux_write(mux_writer, *inputs)
    start_writer_thread(controller, write)


def start_writer_thread(
    controller: MuxWriterController, write: Callable[[bytes], None]
):
    def write_loop():
        while True:
            packet = controller.next_packet()
            write(packet.to_bytes())

    thread = Thread(target=write_loop, daemon=True)
    thread.start()


def mux_write(writer: MuxWriter, *inputs: rx.Observable):
    def write(pair: Tuple[int, Tensor]):
        stream_id, tensor = pair
        tensor_packet = TensorPacket.from_tensor(tensor)
        buf = tensor_packet.to_bytes()
        writer.save(stream_id, buf)

    _rx_mux(*inputs).subscribe(write)


def _rx_mux(*xss: rx.Observable) -> rx.Observable:
    """Combines observables into single observable of indexed tuples.

    ```
    A:   --- A1 -------- A2 -- A3 ----------->
    B:   -------- B1 ----------------- B3 --->
                    [ rx_mux ]
    out: --- A1 - B1 --- A2 -- A3 ---- B3 --->
    ```

    The output events are of type `tuple[int, AOut | BOut]`,
    where the first item represents the stream index (A = 0, B = 1),
    and the second item holds the data.
    """

    def pair_index(i: int) -> Callable[[Any], Any]:
        def inner(x: Any) -> Tuple[int, Any]:
            return i, x

        return inner

    paired = [xs.pipe(ops.map(pair_index(i))) for i, xs in enumerate(xss)]
    return rx.from_iterable(paired).pipe(ops.merge_all())
