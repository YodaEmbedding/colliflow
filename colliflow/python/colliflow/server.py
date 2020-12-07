import asyncio
import json
from asyncio import StreamReader, StreamWriter

from colliflow.model import Model


class Server:
    def __init__(self, host: str = "localhost", port: int = 0):
        self._host = host
        self._port = port

    def start(self):
        asyncio.run(self.start_async())

    async def start_async(self):
        server = await asyncio.start_server(
            self.client_handler, self._host, self._port
        )
        await server.serve_forever()

    async def client_handler(self, reader: StreamReader, writer: StreamWriter):
        """Receives collaborative graph from client and sets it up."""
        print("New client...")
        ip, port = writer.get_extra_info("peername")
        print(f"Connected to {ip}:{port}")

        line = await reader.readline()
        model = Model.deserialize(line.decode())
        print(model)
        await self._model_setup(model, writer)
        model.to_rx()

    async def _model_setup(self, model: Model, writer: StreamWriter):
        async for module_id, result in model.setup():
            response_dict = {"module_id": module_id, "result": result}
            writer.write(f"{json.dumps(response_dict)}\n".encode())
            await writer.drain()


__all__ = [
    "Server",
]
