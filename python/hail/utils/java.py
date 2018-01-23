import SocketServer
import socket
import sys
from threading import Thread

import py4j
from pyspark.sql.utils import CapturedException
from decorator import decorator
import numpy as np


class FatalError(Exception):
    """:class:`.FatalError` is an error thrown by Hail method failures"""


class Env:
    _jvm = None
    _gateway = None
    _hail_package = None
    _jutils = None
    _hc = None
    _counter = 0

    @staticmethod
    def _get_uid():
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
            from hail2 import init
            import sys
            sys.stderr.write("Initializing Spark and Hail with default parameters...\n")
            init()
            assert Env._hc is not None
        return Env._hc


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
    args = [x] if isinstance(x, str) or isinstance(x, unicode) else x
    return jindexed_seq(args)


def jset_args(x):
    args = [x] if isinstance(x, str) or isinstance(x, unicode) else x
    return jset(args)


def jiterable_to_list(it):
    if it:
        return list(Env.jutils().iterableToArrayList(it))
    else:
        return None

def escape_str(s):
    return Env.jutils().escapePyString(s)

def escape_id(s):
    return Env.jutils().escapeIdentifier(s)

def jarray_to_list(a):
    return list(a) if a else None

def numpy_from_breeze(bdm):
    isT = bdm.isTranspose()
    rows, cols = bdm.rows(), bdm.cols()
    entries = rows * cols

    if bdm.offset() != 0:
        raise ValueError("Expected offset of Breeze matrix to be 0, found {}"
                         .format(bdm.offset()))
    expected_stride = cols if isT else rows
    if bdm.majorStride() != expected_stride:
        raise ValueError("Expected major stride of Breeze matrix to be {}, found {}"
                         .format(expected_stride, bdm.majorStride()))
    if entries > 0x7fffffff:
        raise ValueError("rows * cols must be smaller than {}, found {} by {} matrix"
                         .format(0x7fffffff, rows, cols))

    if entries <= 0x100000:
        b = Env.jutils().bdmGetBytes(bdm, 0, entries)
    else:
        b = bytearray()
        i = 0
        while (i < entries):
            n = min(0x100000, entries - i)
            b.extend(Env.jutils().bdmGetBytes(bdm, i, n))
            i += n
    data = np.fromstring(bytes(b), dtype='f8')
    return np.reshape(data, (cols, rows)).T if bdm.isTranspose else np.reshape(data, (rows, cols))


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

@decorator
def handle_py4j(func, *args, **kwargs):
    try:
        r = func(*args, **kwargs)
    except py4j.protocol.Py4JJavaError as e:
        tpl = Env.jutils().handleForPython(e.java_exception)
        deepest, full = tpl._1(), tpl._2()
        raise FatalError('%s\n\nJava stack trace:\n%s\n'
                         'Hail version: %s\n'
                         'Error summary: %s' % (deepest, full, Env.hc().version, deepest))
    except py4j.protocol.Py4JError as e:
        if e.args[0].startswith('An error occurred while calling'):
            msg = 'An error occurred while calling into JVM, probably due to invalid parameter types.'
            raise FatalError('%s\n\nJava stack trace:\n%s\n'
                             'Hail version: %s\n'
                             'Error summary: %s' % (msg, e.message, Env.hc().version, msg))
        else:
            raise e
    except CapturedException as e:
        raise FatalError('%s\n\nJava stack trace:\n%s\n'
                         'Hail version: %s\n'
                         'Error summary: %s' % (e.desc, e.stackTrace, Env.hc().version, e.desc))
    return r


class LoggingTCPHandler(SocketServer.StreamRequestHandler):
    def handle(self):
        for line in self.rfile:
            sys.stderr.write(line)


class SimpleServer(SocketServer.ThreadingMixIn, SocketServer.TCPServer):
    daemon_threads = True
    allow_reuse_address = True

    def __init__(self, server_address, handler_class):
        SocketServer.TCPServer.__init__(self, server_address, handler_class)


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
