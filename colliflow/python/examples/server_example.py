from colliflow.server import Server

from .shared_modules import *

IP = "0.0.0.0"
PORT = 5678


if __name__ == "__main__":
    server = Server(IP, PORT)
    server.start()
