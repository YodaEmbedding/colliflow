import asyncio
import json
from asyncio import StreamReader, StreamWriter
from typing import Any

from colliflow.model import Model
from colliflow.network.connectors import ServersideConnector


class Server:
    def __init__(self, host: str = "localhost", port: int = 0):
        self._host = host
        self._port = port

    def start(self):
        asyncio.run(self.start_async())

    async def start_async(self):
        server = await asyncio.start_server(
            _client_handler, self._host, self._port
        )
        await server.serve_forever()


async def _client_handler(reader: StreamReader, writer: StreamWriter):
    """Receives collaborative graph from client and sets it up."""
    print("New client...")
    ip, port = writer.get_extra_info("peername")
    print(f"Connected to {ip}:{port}")

    connector = ServersideConnector(reader, writer)
    await connector.connect()

    return

    line = await reader.readline()
    model = Model.deserialize(line.decode())
    print(model)
    await _model_setup(model, writer)
    print("model.setup() complete!")
    model.to_rx([])
    print("model.to_rx() complete!")


async def _model_setup(model: Model, writer: StreamWriter):
    async for module_id, result in model.setup():
        response_dict = {"module_id": module_id, "result": result}
        print("sending", response_dict)
        await _writejson(writer, response_dict)


async def _writejson(writer: StreamWriter, obj: Any):
    writer.write(f"{json.dumps(obj)}\n".encode())
    await writer.drain()


__all__ = [
    "Server",
]
