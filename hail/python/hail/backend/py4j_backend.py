import abc
import json

import py4j

import hail
from hail.expr.types import dtype
from hail.expr.table_type import ttable
from hail.expr.matrix_type import tmatrix
from hail.expr.blockmatrix_type import tblockmatrix
from hail.ir import JavaIR
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


class Py4JBackend(Backend):
    def __init__(self):
        super().__init__()

    @abc.abstractmethod
    def jvm(self):
        pass

    @abc.abstractmethod
    def hail_package(self):
        pass

    @abc.abstractmethod
    def utils_package_object(self):
        pass

    # FIXME why is this one different?
    def _parse_value_ir(self, code, ref_map={}):
        return self._jbackend.pyParseValueIR(
            code,
            {k: t._parsable_string() for k, t in ref_map.items()})

    def _parse_table_ir(self, code, ref_map={}):
        return self._jbackend.pyParseTableIR(code, ref_map)

    def _parse_matrix_ir(self, code, ref_map={}):
        return self._jbackend.pyParseMatrixIR(code, ref_map)

    def _parse_blockmatrix_ir(self, code, ref_map={}):
        return self._jbackend.pyParseBlockMatrixIR(code, ref_map)

    def _to_java_ir(self, ir, ref_map, parse):
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
            self._jbackend.removeIR(id)

    def value_type(self, ir):
        jir_id = self._to_java_value_ir(ir)
        return dtype(self._jbackend.pyValueType(jir_id))

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

    def execute(self, ir, timed=False):
        jir_id = self._to_java_value_ir(ir)
        # print(self._hail_package.expr.ir.Pretty.apply(jir, True, -1))
        try:
            result = json.loads(self._jbackend.executeJSON(jir_id))
            value = ir.typ._from_json(result['value'])
            timings = result['timings']

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
