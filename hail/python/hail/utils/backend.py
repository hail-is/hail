import abc

from hail.utils.java import *
from hail.expr.types import HailType
from hail.ir.renderer import Renderer


class Backend(object):
    @abc.abstractmethod
    def interpret(self, ir):
        return


class SparkBackend(Backend):
    def interpret(self, ir):
        assert isinstance(ir, hail.ir.IR)

        r = Renderer(stop_at_jir=False)
        code = r(ir)
        ir_map = {name: ir._jir for name, ir in r.jirs.items()}

        jir = Env.hail().expr.Parser.parse_value_ir(code, {}, ir_map)

        typ = HailType._from_java(jir.typ())
        result = Env.hail().expr.ir.Interpret.interpretPyIR(code, {}, ir_map)

        return typ._from_json(result)
