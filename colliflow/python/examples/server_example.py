import asyncio
from asyncio import StreamReader, StreamWriter

from colliflow import Model
from .shared_modules import *

IP = "0.0.0.0"
PORT = 5678


async def client_handler(reader: StreamReader, writer: StreamWriter):
    print("New client...")
    ip, port = writer.get_extra_info("peername")
    print(f"Connected to {ip}:{port}")

    # submodel = await get_submodel(reader, writer)
    line = await reader.readline()
    # if len(line) == 0:
    #     break
    submodel = Model.deserialize(line)
    print(submodel)

    # TODO what does to_rx mean? No inputs? or outputs?
    #
    # But submodel should contain information on TCP/UDP I/O...
    # 1. Duplex TCP
    # 2. Duplex UDP
    # 3. UDP in, TCP out

    # observable = submodel.to_rx()
    #
    # TODO what do we subscribe on? asyncio_scheduler? does it matter?
    # observable.subscribe()
    # print("subscribed")


async def main():
    server = await asyncio.start_server(client_handler, IP, PORT)
    print("Started server")
    await server.serve_forever()


if __name__ == "__main__":
    asyncio.run(main())


# TODO
# continuously monitor for sockets...
# then establish graph/init/etc
