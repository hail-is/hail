import abc
import http.client
import socket
import socketserver
import sys
from threading import Thread
from typing import Mapping, Optional, Tuple

import orjson
import py4j
import requests
from py4j.java_gateway import JavaObject, JVMView

import hail
from hail.expr import construct_expr
from hail.fs.hadoop_fs import HadoopFS
from hail.ir import JavaIR
from hail.utils.java import Env, FatalError, scala_package_object
from hail.version import __version__
from hailtop.aiocloud.aiogoogle import GCSRequesterPaysConfiguration
from hailtop.utils import sync_retry_transient_errors

from ..hail_logging import Logger
from .backend import ActionTag, Backend, fatal_error_from_java_error_triplet

# This defaults to 65536 and fails if a header is longer than _MAXLINE
# The timing json that we output can exceed 65536 bytes so we raise the limit
http.client._MAXLINE = 2**20


_installed = False
_original = None


def install_exception_handler():
    global _installed
    global _original
    if not _installed:
        _original = py4j.protocol.get_return_value
        _installed = True
        # The original `get_return_value` is not patched, it's idempotent.
        patched = handle_java_exception(_original)
        # only patch the one used in py4j.java_gateway (call Java API)
        py4j.java_gateway.get_return_value = patched


def uninstall_exception_handler():
    global _installed
    if _installed:
        _installed = False
        py4j.protocol.get_return_value = _original


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

            if not Env.is_fully_initialized():
                raise ValueError('Error occurred during Hail initialization.') from e

            tpl = Env.jutils().handleForPython(e.java_exception)
            deepest, full, error_id = tpl._1(), tpl._2(), tpl._3()
            raise fatal_error_from_java_error_triplet(deepest, full, error_id) from None
        except pyspark.sql.utils.CapturedException as e:
            raise FatalError(
                '%s\n\nJava stack trace:\n%s\n'
                'Hail version: %s\n'
                'Error summary: %s' % (e.desc, e.stackTrace, hail.__version__, e.desc)
            ) from None

    return deco


class SimpleServer(socketserver.ThreadingMixIn, socketserver.TCPServer):
    daemon_threads = True
    allow_reuse_address = True

    def __init__(self, server_address, handler_class):
        socketserver.TCPServer.__init__(self, server_address, handler_class)


class LoggingTCPHandler(socketserver.StreamRequestHandler):
    def handle(self):
        for line in self.rfile:
            sys.stderr.write(line.decode("ISO-8859-1"))


class Log4jLogger(Logger):
    def __init__(self, log_pkg):
        self._log_pkg = log_pkg

    def error(self, msg):
        self._log_pkg.error(msg)

    def warning(self, msg):
        self._log_pkg.warn(msg)

    def info(self, msg):
        self._log_pkg.info(msg)


def connect_logger(utils_package_object, host, port):
    """
    This method starts a simple server which listens on a port for a
    client to connect and start writing messages. Whenever a message
    is received, it is written to sys.stderr. The server is run in
    a daemon thread from the caller, which is killed when the caller
    thread dies.

    If the socket is in use, then the server tries to listen on the
    next port (port + 1). After 25 tries, it gives up.

    :param str host: Hostname for server.
    :param int port: Port to listen on.
    """
    server = None
    tries = 0
    max_tries = 25
    while not server:
        try:
            server = SimpleServer((host, port), LoggingTCPHandler)
        except socket.error:
            port += 1
            tries += 1

            if tries >= max_tries:
                sys.stderr.write(
                    'WARNING: Could not find a free port for logger, maximum retries {} exceeded.'.format(max_tries)
                )
                return

    t = Thread(target=server.serve_forever, args=())

    # The thread should be a daemon so that it shuts down when the parent thread is killed
    t.daemon = True

    t.start()
    utils_package_object.addSocketAppender(host, port)


action_routes = {
    ActionTag.VALUE_TYPE: '/value/type',
    ActionTag.TABLE_TYPE: '/table/type',
    ActionTag.MATRIX_TABLE_TYPE: '/matrixtable/type',
    ActionTag.BLOCK_MATRIX_TYPE: '/blockmatrix/type',
    ActionTag.LOAD_REFERENCES_FROM_DATASET: '/references/load',
    ActionTag.FROM_FASTA_FILE: '/references/from_fasta',
    ActionTag.EXECUTE: '/execute',
    ActionTag.PARSE_VCF_METADATA: '/vcf/metadata/parse',
    ActionTag.IMPORT_FAM: '/fam/import',
}


def parse_timings(str: Optional[str]) -> Optional[dict]:
    def parse(node):
        return {
            'name': node[0],
            'total_time': node[1],
            'self_time': node[2],
            'children': [parse(c) for c in node[3]],
        }

    return None if str is None else parse(orjson.loads(str))


class Py4JBackend(Backend):
    @abc.abstractmethod
    def __init__(
        self,
        jvm: JVMView,
        jbackend: JavaObject,
        jhc: JavaObject,
        tmpdir: str,
        remote_tmpdir: str,
    ):
        super().__init__()
        import base64

        def decode_bytearray(encoded):
            return base64.standard_b64decode(encoded)

        # By default, py4j's version of this function does extra
        # work to support python 2. This eliminates that.
        py4j.protocol.decode_bytearray = decode_bytearray

        self._jvm = jvm
        self._hail_package = getattr(self._jvm, 'is').hail
        self._utils_package_object = scala_package_object(self._hail_package.utils)
        self._jhc = jhc

        self._jbackend = self._hail_package.backend.driver.Py4JQueryDriver(jbackend)
        self._jbackend.pySetLocalTmp(tmpdir)
        self._jbackend.pySetRemoteTmp(remote_tmpdir)

        self._jhttp_server = self._jbackend.pyHttpServer()

        self._gcs_requester_pays_config = None
        self._fs = None

        # This has to go after creating the SparkSession. Unclear why.
        # Maybe it does its own patch?
        install_exception_handler()

        jar_version = self._jhc.version()
        if jar_version != __version__:
            raise RuntimeError(
                f"Hail version mismatch between JAR and Python library\n"
                f"  JAR:    {jar_version}\n"
                f"  Python: {__version__}"
            )

    def jvm(self):
        return self._jvm

    def hail_package(self):
        return self._hail_package

    def utils_package_object(self):
        return self._utils_package_object

    @property
    def gcs_requester_pays_configuration(self) -> Optional[GCSRequesterPaysConfiguration]:
        return self._gcs_requester_pays_config

    @gcs_requester_pays_configuration.setter
    def gcs_requester_pays_configuration(self, config: Optional[GCSRequesterPaysConfiguration]):
        self._gcs_requester_pays_config = config
        project, buckets = (None, None) if config is None else (config, None) if isinstance(config, str) else config
        self._jbackend.pySetGcsRequesterPaysConfig(project, buckets)
        self._fs = None  # stale

    @property
    def fs(self):
        if self._fs is None:
            self._fs = HadoopFS(self._utils_package_object, self._jbackend.pyFs())
        return self._fs

    @property
    def logger(self):
        if self._logger is None:
            self._logger = Log4jLogger(self._utils_package_object)
        return self._logger

    def _rpc(self, action, payload) -> Tuple[bytes, Optional[dict]]:
        data = orjson.dumps(payload)
        path = action_routes[action]

        def go():
            port = self._jhttp_server.port()
            try:
                return requests.post(f'http://localhost:{port}{path}', data=data)
            except requests.exceptions.ConnectionError:
                self._stop_jhttp_server()
                self._jhttp_server = self._jbackend.pyHttpServer()
                raise

        resp = sync_retry_transient_errors(go)

        if resp.status_code >= 400:
            error_json = orjson.loads(resp.content)
            raise fatal_error_from_java_error_triplet(
                error_json['short'], error_json['expanded'], error_json['error_id']
            )

        return resp.content, parse_timings(resp.headers.get('X-Hail-Timings', None))

    def persist_expression(self, expr):
        t = expr.dtype
        return construct_expr(JavaIR(t, self._jbackend.pyExecuteLiteral(self._render_ir(expr._ir))), t)

    def set_flags(self, **flags: Mapping[str, str]):
        available = self._jbackend.pyAvailableFlags()
        invalid = []
        for flag, value in flags.items():
            if flag in available:
                self._jbackend.pySetFlag(flag, value)
            else:
                invalid.append(flag)
        if len(invalid) != 0:
            raise FatalError(
                "Flags {} not valid. Valid flags: \n    {}".format(', '.join(invalid), '\n    '.join(available))
            )

    def get_flags(self, *flags) -> Mapping[str, str]:
        return {flag: self._jbackend.pyGetFlag(flag) for flag in flags}

    def _add_reference_to_scala_backend(self, rg):
        self._jbackend.pyAddReference(orjson.dumps(rg._config).decode('utf-8'))

    def _remove_reference_from_scala_backend(self, name):
        self._jbackend.pyRemoveReference(name)

    def add_sequence(self, name, fasta_file, index_file):
        self._jbackend.pyAddSequence(name, fasta_file, index_file)

    def remove_sequence(self, name):
        self._jbackend.pyRemoveSequence(name)

    def add_liftover(self, name, chain_file, dest_reference_genome):
        self._jbackend.pyAddLiftover(name, chain_file, dest_reference_genome)

    def remove_liftover(self, name, dest_reference_genome):
        self._jbackend.pyRemoveLiftover(name, dest_reference_genome)

    def _register_ir_function(self, name, type_parameters, argument_names, argument_types, return_type, code):
        self._registered_ir_function_names.add(name)
        self._jbackend.pyRegisterIR(
            name,
            [ta._parsable_string() for ta in type_parameters],
            argument_names,
            [pt._parsable_string() for pt in argument_types],
            return_type._parsable_string(),
            code,
        )

    def _parse_blockmatrix_ir(self, code):
        return self._jbackend.parse_blockmatrix_ir(code)

    def _to_java_blockmatrix_ir(self, ir):
        return self._parse_blockmatrix_ir(self._render_ir(ir))

    def _stop_jhttp_server(self):
        try:
            self._jhttp_server.close()
        except requests.exceptions.ConnectionError:
            pass

    def stop(self):
        self._stop_jhttp_server()
        self._jbackend.close()
        self._jhc.stop()
        self._jhc = None
        uninstall_exception_handler()
        super().stop()
