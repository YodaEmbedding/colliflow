import asyncio
import json
import socket
import time
from asyncio import StreamReader, StreamWriter
from typing import Any, List, Tuple

import rx
import rx.operators as ops
from rx.scheduler import NewThreadScheduler

from colliflow.model import Model, _rx_to_async_iter
from colliflow.network.sockstream import SocketStreamReader, SocketStreamWriter
from colliflow.serialization.mux_reader import read_mux_packet, start_reader
from colliflow.serialization.mux_writer import start_writer

WAIT_CONNECTION = 0.01


class ClientsideConnector:
    """Open connection to data stream input/output sockets."""

    def __init__(self, graph: Model, host: str, port: int):
        self._graph = graph
        self._host = host
        self._port = port

    def connect(self) -> Tuple[socket.socket, socket.socket]:
        comm_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        comm_sock.connect((self._host, self._port))
        comm_reader = SocketStreamReader(comm_sock)
        comm_writer = SocketStreamWriter(comm_sock)

        comm_writer.writeline(self._graph.serialize().encode())

        sock_info = comm_reader.readjson()
        out_port = sock_info[0]["port"]
        in_port = sock_info[1]["port"]

        time.sleep(WAIT_CONNECTION)

        out_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        out_sock.connect((self._host, out_port))

        in_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        in_sock.connect((self._host, in_port))

        setup_response = comm_reader.readjson()
        model_response = comm_reader.readjson()

        if model_response["status"] != "ready":
            raise RuntimeError("Server could not set up graph.")

        return in_sock, out_sock


class ServersideConnector:
    def __init__(self, reader: StreamReader, writer: StreamWriter):
        self._reader = reader
        self._writer = writer
        self._in_sock_listener = socket.socket(
            socket.AF_INET, socket.SOCK_STREAM
        )
        self._out_sock_listener = socket.socket(
            socket.AF_INET, socket.SOCK_STREAM
        )
        self._in_sock_conn: socket.socket
        self._out_sock_conn: socket.socket
        self._inputs: List[rx.Observable]
        self._outputs: List[rx.Observable]
        self._graph: Model

    async def connect(self):
        await self._receive_model()
        await self._setup_network_io()
        await self._setup_model()
        self._start_network_io()
        await self._start_model()

    async def _receive_model(self):
        line = await self._reader.readline()
        self._graph = Model.deserialize(line.decode())

    async def _setup_network_io(self):
        def accept_conn(item: Tuple[str, socket.socket]) -> Any:
            tag, sock = item
            sock.listen()
            conn, _ = sock.accept()
            return tag, conn

        def accept_conn_rx(item: Tuple[str, socket.socket]) -> Any:
            return rx.from_iterable([item]).pipe(
                ops.observe_on(NewThreadScheduler()),
                ops.map(lambda x: accept_conn(x)),
            )

        self._in_sock_listener.bind(("0.0.0.0", 0))
        self._out_sock_listener.bind(("0.0.0.0", 0))

        _, in_port = self._in_sock_listener.getsockname()
        _, out_port = self._out_sock_listener.getsockname()

        items = [
            ("in", self._in_sock_listener),
            ("out", self._out_sock_listener),
        ]

        connections = rx.from_iterable(items).pipe(
            ops.flat_map(lambda x: accept_conn_rx(x)),
        )

        # NOTE: Possible race condition if threads haven't started listening
        # before client receives message and attempts to connect
        await self._send_obj([{"port": in_port}, {"port": out_port}])

        async for tag, conn in _rx_to_async_iter(connections):
            if tag == "in":
                self._in_sock_conn = conn
            elif tag == "out":
                self._out_sock_conn = conn

    async def _setup_model(self):
        setup_results = {}
        async for module_id, result in self._graph.setup():
            # TODO could send this result immediately... but would drain block?
            setup_results[module_id] = result
        await self._send_obj(setup_results)

    def _start_network_io(self):
        num_input_streams = len(self._graph._inputs)
        write = self._out_sock_conn.sendall
        read = lambda: read_mux_packet(self._in_sock_conn)

        self._inputs = start_reader(num_input_streams, read)
        self._outputs = self._graph.to_rx(*self._inputs)
        start_writer(self._outputs, write)

    async def _start_model(self):
        await asyncio.sleep(WAIT_CONNECTION)
        await self._send_obj({"status": "ready"})

    async def _send_obj(self, obj):
        self._writer.write((json.dumps(obj) + "\n").encode())
        await self._writer.drain()
