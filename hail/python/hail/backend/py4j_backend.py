import abc

import py4j

import hail
from hail.ir.renderer import CSERenderer
from hail.utils.java import FatalError, Env
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

    def register_ir_function(self, name, type_parameters, argument_names, argument_types, return_type, body):
        r = CSERenderer(stop_at_jir=True)
        code = r(body._ir)
        jbody = (self._parse_value_ir(code, ref_map=dict(zip(argument_names, argument_types)), ir_map=r.jirs))

        Env.hail().expr.ir.functions.IRFunctionRegistry.pyRegisterIR(
            name,
            [ta._parsable_string() for ta in type_parameters],
            argument_names, [pt._parsable_string() for pt in argument_types],
            return_type._parsable_string(),
            jbody)
