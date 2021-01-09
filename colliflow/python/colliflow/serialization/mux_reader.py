import socket
from asyncio import IncompleteReadError
from threading import Thread
from typing import Any, Callable, List

import rx
import rx.operators as ops
from rx.scheduler import NewThreadScheduler
from rx.subject import Subject

from .mux_packet import MuxPacket
from .tensor_packet import TensorPacket


def start_reader(
    num_streams: int, read: Callable[[], Any]
) -> List[rx.Observable]:
    mux_packets = Subject()
    start_reader_thread(mux_packets, read)
    return mux_read(mux_packets, num_streams)


def start_reader_thread(subject: Subject, read: Callable[[], Any]):
    def read_loop():
        while True:
            subject.on_next(read())

    thread = Thread(target=read_loop, daemon=True)
    thread.start()


def mux_read(
    mux_packets: rx.Observable, num_streams: int
) -> List[rx.Observable]:
    return [
        mux_packets.pipe(
            ops.observe_on(NewThreadScheduler()),
            ops.filter(lambda x, i=i: x.stream_id == i),
            ops.map(lambda x: x.payload),
            _tensor_stream_deserializer,
        )
        for i in range(num_streams)
    ]


def read_mux_packet(sock: socket.socket) -> MuxPacket:
    stream_id = _readint(sock)
    payload_size = _readint(sock)
    payload = _readexactly(sock, payload_size)
    return MuxPacket(
        stream_id=stream_id, payload_size=payload_size, payload=payload
    )


class TensorStreamDeserializer:
    def __init__(self, subject: Subject):
        self._subject = subject
        self._buffer = bytearray()

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


def _tensor_stream_deserializer(in_stream: rx.Observable) -> rx.Observable:
    out_stream = Subject()
    deserializer = TensorStreamDeserializer(out_stream)
    in_stream.pipe(
        ops.do_action(deserializer.on_next),
    ).subscribe()
    return out_stream


def _readexactly(sock: socket.socket, num_bytes: int) -> bytes:
    buf = bytearray(num_bytes)
    pos = 0
    while pos < num_bytes:
        n = sock.recv_into(memoryview(buf)[pos:])
        if n == 0:
            raise IncompleteReadError(bytes(buf[:pos]), num_bytes)
        pos += n
    return bytes(buf)


def _readint(sock: socket.socket, num_bytes: int = 4) -> int:
    return int.from_bytes(_readexactly(sock, num_bytes), byteorder="big")
