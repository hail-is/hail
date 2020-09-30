import json
import socket
import struct
import logging
from hailtop.utils import sync_retry_transient_errors, TransientError


log = logging.getLogger('query.sockets')


def connect_to_java() -> 'ServiceBackendSocketConnection':
    return ServiceBackendSocketConnection()


class EndOfStream(TransientError):
    pass


class ServiceBackendSocketConnection:
    LOAD_REFERENCES_FROM_DATASET = 1
    VALUE_TYPE = 2
    TABLE_TYPE = 3
    MATRIX_TABLE_TYPE = 4
    BLOCK_MATRIX_TYPE = 5
    REFERENCE_GENOME = 6
    EXECUTE = 7
    FLAGS = 8
    GET_FLAG = 9
    UNSET_FLAG = 10
    SET_FLAG = 11
    ADD_USER = 12
    GOODBYE = 254

    FNAME = '/sock/sock'

    def __init__(self):
        pass

    def __enter__(self) -> 'ServiceBackendSocketConnection':
        self._conn = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)  # pylint: disable=attribute-defined-outside-init
        sync_retry_transient_errors(
            self._conn.connect,
            ServiceBackendSocketConnection.FNAME)
        return self

    def __exit__(self, type, value, traceback):
        self.write_int(ServiceBackendSocketConnection.GOODBYE)
        response = self.read_int()
        assert response == ServiceBackendSocketConnection.GOODBYE, response
        self._conn.close()

    def write_int(self, v: int):
        self._conn.sendall(struct.pack('<i', v))

    def write_long(self, v: int):
        self._conn.sendall(struct.pack('<q', v))

    def write_bytes(self, b: bytes):
        n = len(b)
        self.write_int(n)
        self._conn.sendall(b)

    def write_str(self, s: str):
        self.write_bytes(s.encode('utf-8'))

    def read(self, n) -> bytes:
        b = bytearray()
        left = n
        while left > 0:
            t = self._conn.recv(left)
            if not t:
                log.warning('unexpected EOS, Java violated protocol, this will be retried')
                raise EndOfStream()
            left -= len(t)
            b.extend(t)
        return b

    def read_byte(self) -> int:
        b = self.read(1)
        return b[0]

    def read_bool(self) -> bool:
        return self.read_byte() != 0

    def read_int(self) -> int:
        b = self.read(4)
        return struct.unpack('<i', b)[0]

    def read_long(self) -> int:
        b = self.read(8)
        return struct.unpack('<q', b)[0]

    def read_bytes(self) -> bytes:
        n = self.read_int()
        return self.read(n)

    def read_str(self) -> str:
        b = self.read_bytes()
        return b.decode('utf-8')

    def load_references_from_dataset(self, username: str, session_id: str, billing_project: str, bucket: str, path: str):
        self.write_int(ServiceBackendSocketConnection.LOAD_REFERENCES_FROM_DATASET)
        self.write_str(username)
        self.write_str(session_id)
        self.write_str(billing_project)
        self.write_str(bucket)
        self.write_str(path)
        success = self.read_bool()
        if success:
            s = self.read_str()
            try:
                return json.loads(s)
            except json.decoder.JSONDecodeError as err:
                raise ValueError(f'could not decode {s}') from err
        jstacktrace = self.read_str()
        raise ValueError(jstacktrace)

    def value_type(self, username: str, session_id: str, s: str):
        self.write_int(ServiceBackendSocketConnection.VALUE_TYPE)
        self.write_str(username)
        self.write_str(session_id)
        self.write_str(s)
        success = self.read_bool()
        if success:
            s = self.read_str()
            try:
                return json.loads(s)
            except json.decoder.JSONDecodeError as err:
                raise ValueError(f'could not decode {s}') from err
        jstacktrace = self.read_str()
        raise ValueError(jstacktrace)

    def table_type(self, username: str, session_id: str, s: str):
        self.write_int(ServiceBackendSocketConnection.TABLE_TYPE)
        self.write_str(username)
        self.write_str(session_id)
        self.write_str(s)
        success = self.read_bool()
        if success:
            s = self.read_str()
            try:
                return json.loads(s)
            except json.decoder.JSONDecodeError as err:
                raise ValueError(f'could not decode {s}') from err
        jstacktrace = self.read_str()
        raise ValueError(jstacktrace)

    def matrix_table_type(self, username: str, session_id: str, s: str):
        self.write_int(ServiceBackendSocketConnection.MATRIX_TABLE_TYPE)
        self.write_str(username)
        self.write_str(session_id)
        self.write_str(s)
        success = self.read_bool()
        if success:
            s = self.read_str()
            try:
                return json.loads(s)
            except json.decoder.JSONDecodeError as err:
                raise ValueError(f'could not decode {s}') from err
        jstacktrace = self.read_str()
        raise ValueError(jstacktrace)

    def block_matrix_type(self, username: str, session_id: str, s: str):
        self.write_int(ServiceBackendSocketConnection.BLOCK_MATRIX_TYPE)
        self.write_str(username)
        self.write_str(session_id)
        self.write_str(s)
        success = self.read_bool()
        if success:
            s = self.read_str()
            try:
                return json.loads(s)
            except json.decoder.JSONDecodeError as err:
                raise ValueError(f'could not decode {s}') from err
        jstacktrace = self.read_str()
        raise ValueError(jstacktrace)

    def reference_genome(self, username: str, session_id: str, name: str):
        self.write_int(ServiceBackendSocketConnection.REFERENCE_GENOME)
        self.write_str(username)
        self.write_str(session_id)
        self.write_str(name)
        success = self.read_bool()
        if success:
            s = self.read_str()
            try:
                return json.loads(s)
            except json.decoder.JSONDecodeError as err:
                raise ValueError(f'could not decode {s}') from err
        jstacktrace = self.read_str()
        raise ValueError(jstacktrace)

    def execute(self, username: str, session_id: str, billing_project: str, bucket: str, code: str, token: str):
        self.write_int(ServiceBackendSocketConnection.EXECUTE)
        self.write_str(username)
        self.write_str(session_id)
        self.write_str(billing_project)
        self.write_str(bucket)
        self.write_str(code)
        self.write_str(token)
        success = self.read_bool()
        if success:
            s = self.read_str()
            try:
                return json.loads(s)
            except json.decoder.JSONDecodeError as err:
                raise ValueError(f'could not decode {s}') from err
        jstacktrace = self.read_str()
        raise ValueError(jstacktrace)

    def flags(self):
        self.write_int(ServiceBackendSocketConnection.FLAGS)
        success = self.read_bool()
        if success:
            s = self.read_str()
            try:
                return json.loads(s)
            except json.decoder.JSONDecodeError as err:
                raise ValueError(f'could not decode {s}') from err
        jstacktrace = self.read_str()
        raise ValueError(jstacktrace)

    def get_flag(self, name: str):
        self.write_int(ServiceBackendSocketConnection.GET_FLAG)
        self.write_str(name)
        success = self.read_bool()
        if success:
            s = self.read_str()
            try:
                return json.loads(s)
            except json.decoder.JSONDecodeError as err:
                raise ValueError(f'could not decode {s}') from err
        jstacktrace = self.read_str()
        raise ValueError(jstacktrace)

    def unset_flag(self, name: str):
        self.write_int(ServiceBackendSocketConnection.UNSET_FLAG)
        self.write_str(name)
        success = self.read_bool()
        if success:
            s = self.read_str()
            try:
                return json.loads(s)
            except json.decoder.JSONDecodeError as err:
                raise ValueError(f'could not decode {s}') from err
        jstacktrace = self.read_str()
        raise ValueError(jstacktrace)

    def set_flag(self, name: str, value: str):
        self.write_int(ServiceBackendSocketConnection.SET_FLAG)
        self.write_str(name)
        self.write_str(value)
        success = self.read_bool()
        if success:
            s = self.read_str()
            try:
                return json.loads(s)
            except json.decoder.JSONDecodeError as err:
                raise ValueError(f'could not decode {s}') from err
        jstacktrace = self.read_str()
        raise ValueError(jstacktrace)

    def add_user(self, name: str, gsa_key: str):
        self.write_int(ServiceBackendSocketConnection.ADD_USER)
        self.write_str(name)
        self.write_str(gsa_key)
        success = self.read_bool()
        if success:
            return
        jstacktrace = self.read_str()
        raise ValueError(jstacktrace)
