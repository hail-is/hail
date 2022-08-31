from typing import Mapping, Union, Tuple, List
import abc

import py4j
import py4j.java_gateway

import hail
from hail.expr import construct_expr
from hail.ir import JavaIR, finalize_randomness
from hail.ir.renderer import CSERenderer
from hail.utils.java import FatalError, Env
from .backend import Backend, fatal_error_from_java_error_triplet
from ..expr import Expression
from ..expr.types import HailType


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
            raise fatal_error_from_java_error_triplet(deepest, full, error_id) from None
        except pyspark.sql.utils.CapturedException as e:
            raise FatalError('%s\n\nJava stack trace:\n%s\n'
                             'Hail version: %s\n'
                             'Error summary: %s' % (e.desc, e.stackTrace, hail.__version__, e.desc)) from None

    return deco


class Py4JBackend(Backend):
    _jbackend: py4j.java_gateway.JavaObject

    @abc.abstractmethod
    def __init__(self):
        import base64

        def decode_bytearray(encoded):
            return base64.standard_b64decode(encoded)

        # By default, py4j's version of this function does extra
        # work to support python 2. This eliminates that.
        py4j.protocol.decode_bytearray = decode_bytearray

    @abc.abstractmethod
    def jvm(self):
        pass

    @abc.abstractmethod
    def hail_package(self):
        pass

    @abc.abstractmethod
    def utils_package_object(self):
        pass

    @abc.abstractmethod
    def _parse_value_ir(self, code, ref_map={}, ir_map={}):
        pass

    @abc.abstractmethod
    def _to_java_value_ir(self, ir):
        pass

    def register_ir_function(self,
                             name: str,
                             type_parameters: Union[Tuple[HailType, ...], List[HailType]],
                             value_parameter_names: Union[Tuple[str, ...], List[str]],
                             value_parameter_types: Union[Tuple[HailType, ...], List[HailType]],
                             return_type: HailType,
                             body: Expression):
        r = CSERenderer(stop_at_jir=True)
        code = r(finalize_randomness(body._ir))
        jbody = (self._parse_value_ir(code, ref_map=dict(zip(value_parameter_names, value_parameter_types)), ir_map=r.jirs))

        Env.hail().expr.ir.functions.IRFunctionRegistry.pyRegisterIR(
            name,
            [ta._parsable_string() for ta in type_parameters],
            value_parameter_names,
            [pt._parsable_string() for pt in value_parameter_types],
            return_type._parsable_string(),
            jbody)

    def execute(self, ir, timed=False):
        jir = self._to_java_value_ir(ir)
        stream_codec = '{"name":"StreamBufferSpec"}'
        # print(self._hail_package.expr.ir.Pretty.apply(jir, True, -1))
        try:
            result_tuple = self._jbackend.executeEncode(jir, stream_codec, timed)
            (result, timings) = (result_tuple._1(), result_tuple._2())
            value = ir.typ._from_encoding(result)

            return (value, timings) if timed else value
        except FatalError as e:
            self._handle_fatal_error_from_backend(e, ir)

    async def _async_execute(self, ir, timed=False):
        raise NotImplementedError('no async available in Py4JBackend')

    async def _async_execute_many(self, *irs, timed=False):
        raise NotImplementedError('no async available in Py4JBackend')

    async def _async_get_reference(self, name):
        raise NotImplementedError('no async available in Py4JBackend')

    async def _async_get_references(self, names):
        raise NotImplementedError('no async available in Py4JBackend')

    def persist_expression(self, expr):
        return construct_expr(
            JavaIR(self._jbackend.executeLiteral(self._to_java_value_ir(expr._ir))),
            expr.dtype
        )

    def set_flags(self, **flags: Mapping[str, str]):
        available = self._jbackend.availableFlags()
        invalid = []
        for flag, value in flags.items():
            if flag in available:
                self._jbackend.setFlag(flag, value)
            else:
                invalid.append(flag)
        if len(invalid) != 0:
            raise FatalError("Flags {} not valid. Valid flags: \n    {}"
                             .format(', '.join(invalid), '\n    '.join(available)))

    def get_flags(self, *flags) -> Mapping[str, str]:
        return {flag: self._jbackend.getFlag(flag) for flag in flags}

    @property
    def requires_lowering(self):
        return True
