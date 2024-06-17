import re
import sys
from typing import Literal, Optional, Union

from hailtop.config import ConfigVariable, configuration_of

BackendType = Literal['batch', 'local', 'spark']


def choose_backend(backend: Optional[BackendType] = None) -> BackendType:
    return configuration_of(ConfigVariable.QUERY_BACKEND, backend, 'spark')


class HailUserError(Exception):
    """:class:`.HailUserError` is an error thrown by Hail when the user makes an error."""


class FatalError(Exception):
    """:class:`.FatalError` is an error thrown by Hail method failures"""

    def __init__(self, msg, error_id: int = -1):
        super().__init__(msg)
        self._error_id = error_id

    def maybe_user_error(self, ir) -> Union['FatalError', HailUserError]:
        error_sources = ir.base_search(lambda x: x._error_id == self._error_id)
        if len(error_sources) == 0:
            return self

        better_stack_trace = error_sources[0]._stack_trace
        error_message = str(self)
        message_and_trace = f'{error_message}\n' '------------\n' 'Hail stack trace:\n' f'{better_stack_trace}'
        return HailUserError(message_and_trace)


class Env:
    _jutils = None
    _hc = None
    _counter = 0
    _static_rng_uid = 0

    @staticmethod
    def get_uid(base=None):
        if base:
            str_base = base
        else:
            str_base = ''
        Env._counter += 1
        return f"__uid_{str_base}{Env._counter}"

    @staticmethod
    def hail():
        return Env.py4j_backend('Env.hail').hail_package()

    @staticmethod
    def jutils():
        return Env.py4j_backend('Env.jutils').utils_package_object()

    @staticmethod
    def hc() -> 'hail.context.HailContext':
        if not Env._hc:
            sys.stderr.write("Initializing Hail with default parameters...\n")
            sys.stderr.flush()
            from ..context import init

            init()

        assert Env._hc is not None
        return Env._hc

    @staticmethod
    def is_fully_initialized() -> bool:
        return Env._hc is not None

    @staticmethod
    async def _async_hc() -> 'hail.context.HailContext':
        if not Env._hc:
            sys.stderr.write("Initializing Hail with default parameters...\n")
            sys.stderr.flush()

            backend_name = choose_backend()
            if backend_name == 'service':
                from hail.context import init_batch

                await init_batch()
            else:
                return Env.hc()
        assert Env._hc is not None
        return Env._hc

    @staticmethod
    def backend() -> 'hail.backend.Backend':
        return Env.hc()._backend

    @staticmethod
    def py4j_backend(op):
        from hail.backend.py4j_backend import Py4JBackend

        b = Env.backend()
        if isinstance(b, Py4JBackend):
            return b
        else:
            raise NotImplementedError(f"{b.__class__.__name__} doesn't support {op}, only Py4JBackend")

    @staticmethod
    def spark_backend(op):
        from hail.backend.spark_backend import SparkBackend

        b = Env.backend()
        if isinstance(b, SparkBackend):
            return b
        else:
            raise NotImplementedError(f"{b.__class__.__name__} doesn't support {op}, only SparkBackend")

    @staticmethod
    def fs():
        return Env.backend().fs

    @staticmethod
    def spark_session():
        return Env.backend()._spark_session

    _dummy_table = None

    @staticmethod
    def dummy_table():
        if Env._dummy_table is None:
            import hail

            Env._dummy_table = hail.utils.range_table(1, 1).key_by().cache()
        return Env._dummy_table

    @staticmethod
    def next_static_rng_uid():
        result = Env._static_rng_uid
        assert result <= 0x7FFF_FFFF_FFFF_FFFF
        Env._static_rng_uid += 1
        return result

    @staticmethod
    def reset_global_randomness():
        Env._static_rng_uid = 0


def scala_object(jpackage, name):
    return getattr(getattr(jpackage, name + '$'), 'MODULE$')


def scala_package_object(jpackage):
    return scala_object(jpackage, 'package')


def jindexed_seq(x):
    return Env.jutils().arrayListToISeq(x)


def jindexed_seq_args(x):
    args = [x] if isinstance(x, str) else x
    return jindexed_seq(args)


def jiterable_to_list(it):
    if it is not None:
        return list(Env.jutils().iterableToArrayList(it))
    else:
        return None


_parsable_str = re.compile(r'[_a-zA-Z][\w_]*')


def escape_parsable(s):
    if _parsable_str.fullmatch(s):
        return s
    else:
        return '`' + s.encode('unicode_escape').decode('utf-8').replace('`', '\\`') + '`'


def unescape_parsable(s):
    return bytes(s.replace('\\`', '`'), 'utf-8').decode('unicode_escape')


def jarray_to_list(a):
    return list(a) if a else None


def error(msg):
    Env.backend().logger.error(msg)


def warning(msg):
    Env.backend().logger.warning(msg)


def info(msg):
    Env.backend().logger.info(msg)
