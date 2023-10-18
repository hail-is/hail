from typing import Optional, Union, Tuple, List, Set
import os
import socket
import socketserver
import sys
from threading import Thread

import orjson
import py4j
from py4j.java_gateway import JavaGateway, GatewayParameters, launch_gateway

from hail.utils.java import scala_package_object
from hail.ir.renderer import CSERenderer
from hail.ir import finalize_randomness
from .py4j_backend import Py4JBackend, handle_java_exception, action_routes
from .backend import local_jar_information, fatal_error_from_java_error_triplet
from ..hail_logging import Logger
from ..expr import Expression
from ..expr.types import HailType

from hailtop.utils import find_spark_home
from hailtop.fs.router_fs import RouterFS
from hailtop.aiotools.validators import validate_file


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
    global _original
    if _installed:
        _installed = False
        py4j.protocol.get_return_value = _original


class LoggingTCPHandler(socketserver.StreamRequestHandler):
    def handle(self):
        for line in self.rfile:
            sys.stderr.write(line.decode("ISO-8859-1"))


class SimpleServer(socketserver.ThreadingMixIn, socketserver.TCPServer):
    daemon_threads = True
    allow_reuse_address = True

    def __init__(self, server_address, handler_class):
        socketserver.TCPServer.__init__(self, server_address, handler_class)


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
                    'WARNING: Could not find a free port for logger, maximum retries {} exceeded.'.format(max_tries))
                return

    t = Thread(target=server.serve_forever, args=())

    # The thread should be a daemon so that it shuts down when the parent thread is killed
    t.daemon = True

    t.start()
    utils_package_object.addSocketAppender(host, port)


class Log4jLogger(Logger):
    def __init__(self, log_pkg):
        self._log_pkg = log_pkg

    def error(self, msg):
        self._log_pkg.error(msg)

    def warning(self, msg):
        self._log_pkg.warn(msg)

    def info(self, msg):
        self._log_pkg.info(msg)


class LocalBackend(Py4JBackend):
    def __init__(self, tmpdir, log, quiet, append, branching_factor,
                 skip_logging_configuration, optimizer_iterations,
                 jvm_heap_size,
                 *,
                 gcs_requester_pays_project: Optional[str] = None,
                 gcs_requester_pays_buckets: Optional[str] = None
                 ):
        super(LocalBackend, self).__init__()
        assert gcs_requester_pays_project is not None or gcs_requester_pays_buckets is None

        spark_home = find_spark_home()
        hail_jar_path = os.environ.get('HAIL_JAR')
        if hail_jar_path is None:
            try:
                local_jar_info = local_jar_information()
                hail_jar_path = local_jar_info.path
                extra_classpath = ':'.join([f'{spark_home}/jars/*', hail_jar_path, *local_jar_info.extra_classpath])
            except ValueError:
                raise RuntimeError('local backend requires a packaged jar or HAIL_JAR to be set')
        else:
            extra_classpath = ':'.join([f'{spark_home}/jars/*', hail_jar_path])

        jvm_opts = []
        if jvm_heap_size is not None:
            jvm_opts.append(f'-Xmx{jvm_heap_size}')

        port = launch_gateway(
            redirect_stdout=sys.stdout,
            redirect_stderr=sys.stderr,
            java_path=None,
            javaopts=jvm_opts,
            jarpath=f'{spark_home}/jars/py4j-0.10.9.5.jar',
            classpath=extra_classpath,
            die_on_exit=True)
        self._gateway = JavaGateway(
            gateway_parameters=GatewayParameters(port=port, auto_convert=True))
        self._jvm = self._gateway.jvm

        hail_package = getattr(self._jvm, 'is').hail

        self._hail_package = hail_package
        self._utils_package_object = scala_package_object(hail_package.utils)

        self._jbackend = hail_package.backend.local.LocalBackend.apply(
            tmpdir,
            gcs_requester_pays_project,
            gcs_requester_pays_buckets,
            log,
            True,
            append,
            skip_logging_configuration
        )
        self._jhc = hail_package.HailContext.apply(
            self._jbackend, branching_factor, optimizer_iterations, True)

        self._backend_server = hail_package.backend.BackendServer.apply(self._jbackend)
        self._backend_server_port: int = self._backend_server.port()
        self._backend_server.start()
        self._registered_ir_function_names: Set[str] = set()

        # This has to go after creating the SparkSession. Unclear why.
        # Maybe it does its own patch?
        install_exception_handler()

        from hail.context import version

        py_version = version()
        jar_version = self._jhc.version()
        if jar_version != py_version:
            raise RuntimeError(f"Hail version mismatch between JAR and Python library\n"
                               f"  JAR:    {jar_version}\n"
                               f"  Python: {py_version}")

        self._fs = RouterFS()
        self._logger = None

        self._initialize_flags({})

    def validate_file(self, uri: str) -> None:
        validate_file(uri, self._fs.afs)

    def jvm(self):
        return self._jvm

    def hail_package(self):
        return self._hail_package

    def utils_package_object(self):
        return self._utils_package_object

    def register_ir_function(self,
                             name: str,
                             type_parameters: Union[Tuple[HailType, ...], List[HailType]],
                             value_parameter_names: Union[Tuple[str, ...], List[str]],
                             value_parameter_types: Union[Tuple[HailType, ...], List[HailType]],
                             return_type: HailType,
                             body: Expression):
        r = CSERenderer()
        code = r(finalize_randomness(body._ir))
        jbody = self._parse_value_ir(code, ref_map=dict(zip(value_parameter_names, value_parameter_types)))
        self._registered_ir_function_names.add(name)

        self.hail_package().expr.ir.functions.IRFunctionRegistry.pyRegisterIR(
            name,
            [ta._parsable_string() for ta in type_parameters],
            value_parameter_names,
            [pt._parsable_string() for pt in value_parameter_types],
            return_type._parsable_string(),
            jbody)

    def _is_registered_ir_function_name(self, name: str) -> bool:
        return name in self._registered_ir_function_names

    def _rpc(self, action, payload) -> Tuple[bytes, str]:
        data = orjson.dumps(payload)
        path = action_routes[action]
        port = self._backend_server_port
        resp = self._requests_session.post(f'http://localhost:{port}{path}', data=data)
        if resp.status_code >= 400:
            error_json = orjson.loads(resp.content)
            raise fatal_error_from_java_error_triplet(error_json['short'], error_json['expanded'], error_json['error_id'])
        return resp.content, resp.headers.get('X-Hail-Timings', '')

    def stop(self):
        self._backend_server.stop()
        self._jhc.stop()
        self._jhc = None
        self._gateway.shutdown()
        self._registered_ir_function_names = set()
        uninstall_exception_handler()

    @property
    def logger(self):
        if self._logger is None:
            self._logger = Log4jLogger(self._utils_package_object)
        return self._logger

    @property
    def fs(self):
        return self._fs

    @property
    def requires_lowering(self):
        return True
