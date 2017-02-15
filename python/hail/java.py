from py4j.protocol import Py4JJavaError

class FatalError(Exception):
    """:class:`.FatalError` is an error thrown by Hail method failures"""

    def __init__(self, message, java_exception):
        self.msg = message
        self.java_exception = java_exception
        super(FatalError)

    def __str__(self):
        return self.msg


class Env:
    _jvm = None
    _gateway = None
    _hail_package = None
    _jutils = None

    @property
    def jvm(self):
        if not Env._jvm:
            raise EnvironmentError('no Hail context initialized, create one first')
        return Env._jvm

    @property
    def hail(self):
        if not Env._hail_package:
            Env._hail_package = getattr(self.jvm, 'is').hail
        return Env._hail_package

    @property
    def gateway(self):
        if not Env._gateway:
            raise EnvironmentError('no Hail context initialized, create one first')
        return Env._gateway

    @property
    def jutils(self):
        if not Env._jutils:
            Env._jutils = scala_package_object(self.hail.utils)
        return Env._jutils


env = Env()


def jarray(jtype, lst):
    jarr = env.gateway.new_array(jtype, len(lst))
    for i, s in enumerate(lst):
        jarr[i] = s
    return jarr


def scala_object(jpackage, name):
    return getattr(getattr(jpackage, name + '$'), 'MODULE$')


def scala_package_object(jpackage):
    return scala_object(jpackage, 'package')


def jnone():
    return scala_object(env.jvm.scala, 'None')


def jsome(x):
    return env.jvm.scala.Some(x)


def joption(x):
    return jsome(x) if x else jnone()


def from_option(x):
    return x.get() if x.isDefined() else None

def jindexed_seq(x):
    return env.jutils.arrayListToISeq(x)

def jiterable_to_list(it):
    if it:
        return list(env.jutils.iterableToArrayList(it))
    else:
        return None

def handle_py4j(func):
    def wrapper(*args, **kwargs):
        try:
            r = func(*args, **kwargs)
        except Py4JJavaError as e:
            msg = env.jutils.getMinimalMessage(e.java_exception)
            raise FatalError(msg, e.java_exception)

        return r

    return wrapper