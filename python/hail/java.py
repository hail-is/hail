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


def raise_py4j_exception(self, e):
    msg = scala_package_object(self._hail.utils).getMinimalMessage(e.java_exception)
    raise FatalError(msg, e.java_exception)


class Env:
    _jvm = None
    _gateway = None

    @staticmethod
    def jvm():
        if not Env._jvm:
            raise EnvironmentError('no Hail context initialized, create one first')
        return Env._jvm

    @staticmethod
    def hail_package():
        return getattr(Env.jvm(), 'is').hail

    @staticmethod
    def gateway():
        if not Env._gateway:
            raise EnvironmentError('no Hail context initialized, create one first')
        return Env._gateway
