from typing import Dict
import os
import json
import secrets
import socket
import struct

import py4j
import py4j.java_gateway

import hail
from hail.utils.java import scala_package_object
from hail.expr.types import dtype, HailType
from hail.expr.table_type import ttable
from hail.expr.matrix_type import tmatrix
from hail.expr.blockmatrix_type import tblockmatrix
from hail.ir import BaseIR, IR, JavaIR
from hail.ir.renderer import CSERenderer
from hail.utils.java import FatalError, Env, HailUserError
from .backend import Backend


def handle_java_exception(f):
    def deco(*args, **kwargs):
        import pyspark
        try:
            return f(*args, **kwargs)
        except py4j.protocol.Py4JJavaError as e:
            s = e.java_exception.toString()

            # py4j catches NoSuchElementExceptions to stop array iteration
            if s.startswith('java.util.NoSuchElementException'):
                raise

            tpl = Env.jutils().handleForPython(e.java_exception)
            deepest, full, error_id = tpl._1(), tpl._2(), tpl._3()

            if error_id != -1:
                raise FatalError('Error summary: %s' % (deepest,), error_id) from None
            else:
                raise FatalError('%s\n\nJava stack trace:\n%s\n'
                                 'Hail version: %s\n'
                                 'Error summary: %s' % (deepest, full, hail.__version__, deepest), error_id) from None
        except pyspark.sql.utils.CapturedException as e:
            raise FatalError('%s\n\nJava stack trace:\n%s\n'
                             'Hail version: %s\n'
                             'Error summary: %s' % (e.desc, e.stackTrace, hail.__version__, e.desc)) from None

    return deco


class EndOfStream(Exception):
    pass


class UNIXSocketConnection:
    PARSE_VALUE_IR = 1
    VALUE_TYPE = 2
    EXECUTE = 3
    REMOVE_IR = 4
    NOOP = 5

    def __init__(self, jbackend):
        self._jbackend = jbackend

        token = secrets.token_hex(16)
        address = f'/tmp/hail.uds.{token}'
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        sock.bind(address)
        sock.listen(1)
        jsock = jbackend.connectUNIXSocket(address)
        conn, _ = sock.accept()
        jbackend.startUNIXSocketThread(jsock)
        os.unlink(address)
        self._conn = conn

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

    def close(self):
        self._conn.close()

    def parse_value_ir(self, ir_str: str, type_env_str: str) -> int:
        self.write_int(self.PARSE_VALUE_IR)
        self.write_str(ir_str)
        self.write_str(type_env_str)
        succeeded = self.read_bool()
        if not succeeded:
            self._jbackend.reraiseSavedException()
        return self.read_long()

    def value_type(self, id: int) -> str:
        self.write_int(self.VALUE_TYPE)
        self.write_long(id)
        succeeded = self.read_bool()
        if not succeeded:
            self._jbackend.reraiseSavedException()
        return self.read_str()

    def execute(self, id: int) -> str:
        self.write_int(self.EXECUTE)
        self.write_long(id)
        succeeded = self.read_bool()
        if not succeeded:
            self._jbackend.reraiseSavedException()
        return self.read_str()

    def remove_ir(self, id: int):
        self.write_int(self.REMOVE_IR)
        self.write_long(id)
        succeeded = self.read_bool()
        if not succeeded:
            self._jbackend.reraiseSavedException()


class Py4JBackend(Backend):
    def __init__(self, gateway: py4j.java_gateway.JavaGateway, jbackend: py4j.java_gateway.JavaObject):
        super().__init__()
        self._gateway = gateway
        self._jvm = gateway.jvm

        hail_package = getattr(self._jvm, 'is').hail

        self._hail_package = hail_package
        self._utils_package_object = scala_package_object(hail_package.utils)

        self._jbackend = jbackend
        self._conn = UNIXSocketConnection(jbackend)

    def jvm(self) -> py4j.java_gateway.JVMView:
        return self._jvm

    def hail_package(self):
        return self._hail_package

    def utils_package_object(self):
        return self._utils_package_object

    # FIXME why is this one different?
    def _parse_value_ir(self, code, ref_map={}):
        return self._conn.parse_value_ir(
            code,
            json.dumps({k: t._parsable_string() for k, t in ref_map.items()}))

    def _parse_table_ir(self, code, ref_map={}):
        return self._jbackend.pyParseTableIR(code, ref_map)

    def _parse_matrix_ir(self, code, ref_map={}):
        return self._jbackend.pyParseMatrixIR(code, ref_map)

    def _parse_blockmatrix_ir(self, code, ref_map={}):
        return self._jbackend.pyParseBlockMatrixIR(code, ref_map)

    def _to_java_ir(self, ir: BaseIR, ref_map: Dict[str, HailType], parse):
        if ir._jir_id is None:
            r = CSERenderer(stop_at_jir=True)
            # FIXME parse should be static
            ir._jir_id = parse(r(ir), ref_map)
            ir._backend = self
        return ir._jir_id

    def _to_java_value_ir(self, ir, ref_map={}):
        return self._to_java_ir(ir, ref_map, self._parse_value_ir)

    def _to_java_table_ir(self, ir, ref_map={}):
        return self._to_java_ir(ir, ref_map, self._parse_table_ir)

    def _to_java_matrix_ir(self, ir, ref_map={}):
        return self._to_java_ir(ir, ref_map, self._parse_matrix_ir)

    def _to_java_blockmatrix_ir(self, ir, ref_map={}):
        return self._to_java_ir(ir, ref_map, self._parse_blockmatrix_ir)

    def unlink_ir(self, id: int):
        if self._running:
            self._conn.remove_ir(id)

    def value_type(self, ir):
        jir_id = self._to_java_value_ir(ir)
        return dtype(self._conn.value_type(jir_id))

    def table_type(self, tir):
        jir_id = self._to_java_table_ir(tir)
        return ttable._from_json(json.loads(self._jbackend.pyTableType(jir_id)))

    def matrix_type(self, mir):
        jir_id = self._to_java_matrix_ir(mir)
        return tmatrix._from_json(json.loads(self._jbackend.pyMatrixType(jir_id)))

    def blockmatrix_type(self, bmir):
        jir_id = self._to_java_blockmatrix_ir(bmir)
        return tblockmatrix._from_json(json.loads(self._jbackend.pyBlockMatrixType(jir_id)))

    def register_ir_function(self, name, type_parameters, argument_names, argument_types, return_type, body):
        body_jir_id = self._to_java_value_ir(body._ir, ref_map=dict(zip(argument_names, argument_types)))

        Env.hail().expr.ir.functions.IRFunctionRegistry.pyRegisterIR(
            name,
            [ta._parsable_string() for ta in type_parameters],
            argument_names, [pt._parsable_string() for pt in argument_types],
            return_type._parsable_string(),
            body_jir_id)

    def execute(self, ir: IR, *, timed: bool = False, raw: bool = False):
        jir_id = self._to_java_value_ir(ir)
        # print(self._hail_package.expr.ir.Pretty.apply(jir, True, -1))
        try:
            result = json.loads(self._conn.execute(jir_id))

            timings = result['timings']
            value = json.loads(result['value'])
            if not raw:
                value = ir.typ._convert_from_json_na(value)

            return (value, timings) if timed else value
        except FatalError as e:
            error_id = e._error_id

            def criteria(hail_ir):
                return hail_ir._error_id is not None and hail_ir._error_id == error_id

            error_sources = ir.base_search(criteria)
            better_stack_trace = None
            if error_sources:
                better_stack_trace = error_sources[0]._stack_trace

            if better_stack_trace:
                error_message = str(e)
                message_and_trace = (f'{error_message}\n'
                                     '------------\n'
                                     'Hail stack trace:\n'
                                     f'{better_stack_trace}')
                raise HailUserError(message_and_trace) from None

            raise e

    def persist_ir(self, ir):
        return JavaIR(self._jbackend.executeLiteral(self._to_java_value_ir(ir)), self)
