import abc

from hail.ir.renderer import CSERenderer
from hail.utils.java import Env
from .backend import Backend


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
