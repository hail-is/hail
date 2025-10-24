import abc
import glob
import http.client
import logging
import socket
import socketserver
import sys
import warnings
from threading import Thread
from typing import Mapping, Optional, Tuple

import orjson
import py4j
import requests
from py4j.java_gateway import (
    GatewayParameters,
    JavaGateway,
    JavaObject,
    JVMView,
    launch_gateway,
)

from hail.expr import construct_expr
from hail.ir import CSERenderer, JavaIR
from hail.utils.java import Env, FatalError, scala_package_object
from hail.version import __version__
from hailtop.utils import find_spark_home, sync_retry_transient_errors

from ..hail_logging import Logger
from .backend import ActionTag, Backend, fatal_error_from_java_error_triplet, local_jar_information

# This defaults to 65536 and fails if a header is longer than _MAXLINE
# The timing json that we output can exceed 65536 bytes so we raise the limit
http.client._MAXLINE = 2**20


_installed = False
_original = None


def start_py4j_gateway(*, max_heap_size: str | None = None) -> JavaGateway:
    spark_home = find_spark_home()
    _, hail_jar_path, extra_classpath = local_jar_information()
    extra_classpath = ':'.join([f'{spark_home}/jars/*', hail_jar_path, *extra_classpath])

    jvm_opts = []
    if max_heap_size is not None:
        jvm_opts.append(f'-Xmx{max_heap_size}')

    py4j_jars = glob.glob(f'{spark_home}/jars/py4j-*.jar')
    if len(py4j_jars) == 0:
        raise ValueError(f'No py4j JAR found in {spark_home}/jars')

    if len(py4j_jars) > 1:
        logging.warning(f'found multiple p4yj jars arbitrarily choosing the first one: {py4j_jars}')

    port = launch_gateway(
        redirect_stdout=sys.stdout,
        redirect_stderr=sys.stderr,
        java_path=None,
        javaopts=jvm_opts,
        jarpath=py4j_jars[0],
        classpath=extra_classpath,
        die_on_exit=True,
    )

    return JavaGateway(gateway_parameters=GatewayParameters(port=port, auto_convert=True))


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
                'Error summary: %s' % (e.desc, e.stackTrace, __version__, e.desc)
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
        tmpdir: str,
        remote_tmpdir: str,
    ):
        super().__init__()
        import base64

        # By default, py4j's version of this function does extra
        # work to support python 2. This eliminates that.
        py4j.protocol.decode_bytearray = base64.standard_b64decode

        self._jvm = jvm
        self._hail_package = getattr(self._jvm, 'is').hail
        self._utils_package_object = scala_package_object(self._hail_package.utils)

        self._jbackend = self._hail_package.backend.driver.Py4JQueryDriver(jbackend)
        self.local_tmpdir = tmpdir
        self.remote_tmpdir = remote_tmpdir

        self._jhttp_server = self._jbackend.pyHttpServer()
        self._gcs_requester_pays_config = None
        self._logger = Log4jLogger(self._utils_package_object)

        # This has to go after creating the SparkSession. Unclear why.
        # Maybe it does its own patch?
        install_exception_handler()

        feature, *rest = jvm.System.getProperty('java.version').split('.')
        if feature != '11':
            warnings.warn(
                message=(
                    'Hail was built and tested with Java 11. '
                    f'You are using Java {".".join([feature, *rest])} which is not supported. '
                    'This may lead to errors. '
                    'Consider installing Java 11 and setting the JAVA_HOME environment variable.'
                )
            )

        jar_version = scala_package_object(self._hail_package).HAIL_PRETTY_VERSION()
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
    def logger(self) -> Logger:
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

    def register_ir_function(self, name, type_parameters, argument_names, argument_types, return_type, body):
        r = CSERenderer()
        assert not body._ir.uses_randomness
        code = r(body._ir)
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
        uninstall_exception_handler()
        super().stop()

    @property
    def local_tmpdir(self) -> str:
        return self._jbackend.pyGetLocalTmp()

    @local_tmpdir.setter
    def local_tmpdir(self, tmpdir: str) -> None:
        self._jbackend.pySetLocalTmp(tmpdir)

    @property
    def remote_tmpdir(self) -> str:
        return self._jbackend.pyGetRemoteTmp()

    @remote_tmpdir.setter
    def remote_tmpdir(self, tmpdir: str) -> None:
        self._jbackend.pySetRemoteTmp(tmpdir)
