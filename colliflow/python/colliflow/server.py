import asyncio
from asyncio import StreamReader, StreamWriter

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


__all__ = [
    "Server",
]
