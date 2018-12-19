import socketserver
import socket
import sys
import re
from threading import Thread

import py4j
import hail

class FatalError(Exception):
    """:class:`.FatalError` is an error thrown by Hail method failures"""


class Env:
    _jvm = None
    _gateway = None
    _hail_package = None
    _jutils = None
    _hc = None
    _counter = 0
    _seed_generator = None

    @staticmethod
    def get_uid():
        Env._counter += 1
        return "__uid_{}".format(Env._counter)

    @staticmethod
    def jvm():
        if not Env._jvm:
            Env.hc()
            assert Env._jvm is not None
        return Env._jvm

    @staticmethod
    def hail():
        if not Env._hail_package:
            Env._hail_package = getattr(Env.jvm(), 'is').hail
        return Env._hail_package

    @staticmethod
    def gateway():
        if not Env._gateway:
            Env.hc()
            assert Env._gateway is not None
        return Env._gateway

    @staticmethod
    def jutils():
        if not Env._jutils:
            Env._jutils = scala_package_object(Env.hail().utils)
        return Env._jutils

    @staticmethod
    def hc():
        if not Env._hc:
            from hail.context import init
            import sys
            sys.stderr.write("Initializing Spark and Hail with default parameters...\n")
            init()
            assert Env._hc is not None
        return Env._hc

    @staticmethod
    def sql_context():
        return Env.hc()._sql_context

    _dummy_table = None

    @staticmethod
    def dummy_table():
        if Env._dummy_table is None:
            import hail
            Env._dummy_table = hail.utils.range_table(1, 1).key_by().cache()
        return Env._dummy_table

    @staticmethod
    def set_seed(seed):
        Env._seed_generator = hail.utils.HailSeedGenerator(seed)

    @staticmethod
    def next_seed():
        if Env._seed_generator is None:
            Env.set_seed(None)
        return Env._seed_generator.next_seed()


def jarray(jtype, lst):
    jarr = Env.gateway().new_array(jtype, len(lst))
    for i, s in enumerate(lst):
        jarr[i] = s
    return jarr


def scala_object(jpackage, name):
    return getattr(getattr(jpackage, name + '$'), 'MODULE$')


def scala_package_object(jpackage):
    return scala_object(jpackage, 'package')


def jnone():
    return scala_object(Env.jvm().scala, 'None')


def jsome(x):
    return Env.jvm().scala.Some(x)


def joption(x):
    return jsome(x) if x else jnone()


def from_option(x):
    return x.get() if x.isDefined() else None


def jindexed_seq(x):
    return Env.jutils().arrayListToISeq(x)


def jset(x):
    return Env.jutils().arrayListToSet(x)


def jindexed_seq_args(x):
    args = [x] if isinstance(x, str) else x
    return jindexed_seq(args)


def jset_args(x):
    args = [x] if isinstance(x, str) else x
    return jset(args)


def jiterable_to_list(it):
    if it is not None:
        return list(Env.jutils().iterableToArrayList(it))
    else:
        return None


def escape_str(s):
    return Env.jutils().escapePyString(s)

def parsable_strings(strs):
    strs = ' '.join(f'"{escape_str(s)}"' for s in strs)
    return f"({strs})"


_parsable_str = re.compile(r'[\w_]+')


def escape_parsable(s):
    if _parsable_str.fullmatch(s):
        return s
    else:
        return '`' + s.encode('unicode_escape').decode('utf-8').replace('`', '\\`') + '`'


def unescape_parsable(s):
    return bytes(s.replace('\\`', '`'), 'utf-8').decode('unicode_escape')


def escape_id(s):
    return Env.jutils().escapeIdentifier(s)

def jarray_to_list(a):
    return list(a) if a else None

class Log4jLogger:
    log_pkg = None

    @staticmethod
    def get():
        if Log4jLogger.log_pkg is None:
            Log4jLogger.log_pkg = Env.jutils()
        return Log4jLogger.log_pkg


def error(msg):
    Log4jLogger.get().error(msg)


def warn(msg):
    Log4jLogger.get().warn(msg)


def info(msg):
    Log4jLogger.get().info(msg)


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
            deepest, full = tpl._1(), tpl._2()
            raise FatalError('%s\n\nJava stack trace:\n%s\n'
                             'Hail version: %s\n'
                             'Error summary: %s' % (deepest, full, hail.__version__, deepest)) from None
        except pyspark.sql.utils.CapturedException as e:
            raise FatalError('%s\n\nJava stack trace:\n%s\n'
                             'Hail version: %s\n'
                             'Error summary: %s' % (e.desc, e.stackTrace, hail.__version__, e.desc)) from None

    return deco


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


def connect_logger(host, port):
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
    Env.jutils().addSocketAppender(host, port)
