import os
import pickle
import socket
import tempfile
from typing import Union
from select import select

import numpy as np


class AsyncUtilError(ConnectionError):
    def __init__(self, *args: object):
        super().__init__(args)


class SocketOps:
    def __init__(self):
        self.connection: socket.socket = None
        self.header = self.makebuf([1], 'int32')
        self.recvbuf = bytearray()

    def recv(self, buf, sz=None):
        # python memoryview still remembers original shape. flatten it.
        buf = memoryview(buf).cast('b')
        if sz is None:
            sz = len(buf)
        while True:
            actual = self.connection.recv_into(buf, sz)
            if actual == sz:
                break
            if actual == 0:
                raise AsyncUtilError("nothing received")
            sz -= actual
            buf = buf[actual:]
        buf.release()

    def send(self, buf):
        buf = memoryview(buf).cast('b')
        sz = len(buf)
        if sz <= 0:
            return
        try:
            while True:
                actual = self.connection.send(buf)
                if actual == sz:
                    break
                if actual <= 0:
                    raise AsyncUtilError("send failure")
                sz -= actual
                buf = buf[actual:]
        finally:
            buf.release()

    def send_object(self, obj):
        buf = pickle.dumps(obj)
        self.header[0] = len(buf)
        self.send(self.header)
        self.send(buf)

    def recv_object(self):
        self.recv(self.header)
        n = self.header[0]
        if len(self.recvbuf) < n:
            self.recvbuf = self.recvbuf.zfill(n)
        self.recv(self.recvbuf, n)
        return pickle.loads(self.recvbuf[:n])

    def makebuf(self, shape, type: Union[str, int]):
        if type == 4:
            return np.empty(shape=shape, dtype='float32')
        elif type == 8:
            return np.empty(shape=shape, dtype='float64')
        elif isinstance(type, str):
            return np.empty(shape=shape, dtype=type)
        else:
            raise Exception(f"unrecognized type {type}")


class AsyncServer(SocketOps):
    def __init__(self, type: str):
        super().__init__()
        self.type = type
        self.socket_dir = tempfile.mkdtemp(prefix=f"ai2bmd-{type}-")
        self.socket_path = os.path.join(self.socket_dir, "socket")
        self.server_socket = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.server_socket.bind(self.socket_path)
        self.server_socket.listen()
        self.connection = None
        self.socket_client_address = None

    def accept(self):
        self.connection, self.socket_client_address = self.server_socket.accept()

    def close(self):
        self.connection.close()
        self.server_socket.close()

    def wait_for_data(self, timeout):
        rl, _, _ = select([self.connection], [], [], timeout)
        return len(rl) == 1



class AsyncClient(SocketOps):
    def __init__(self, socket_path):
        super().__init__()
        self.socket_path = socket_path
        self.connection = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.connection.connect(socket_path)
