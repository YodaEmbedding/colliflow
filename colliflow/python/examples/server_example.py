import asyncio
from asyncio import StreamReader, StreamWriter

from colliflow import Model, Server

from .shared_modules import *

IP = "0.0.0.0"
PORT = 5678


async def client_handler(reader: StreamReader, writer: StreamWriter):
    print("New client...")
    ip, port = writer.get_extra_info("peername")
    print(f"Connected to {ip}:{port}")

    line = await reader.readline()
    submodel = model_from_config(line.decode())
    print(submodel)

    observables = submodel.to_rx([])
    observable = observables[0]

    # TODO subscribe on? asyncio_scheduler? NewThreadScheduler? does it matter?
    observable.subscribe()
    print("subscribed")


async def main():
    server = await asyncio.start_server(client_handler, IP, PORT)
    print("Started server")
    await server.serve_forever()


if __name__ == "__main__":
    server = Server(IP, PORT)
    server.start()
