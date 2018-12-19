import abc

from hail.utils.java import *
from hail.expr.types import dtype
from hail.expr.table_type import *
from hail.expr.matrix_type import *
from hail.ir.renderer import Renderer


class Backend(object):
    @abc.abstractmethod
    def execute(self, ir):
        return

    # FIXME delete
    def interpret(self, ir):
        return self.execute(ir)

    @abc.abstractmethod
    def table_read_type(self, table_read_ir):
        return

    @abc.abstractmethod
    def matrix_read_type(self, matrix_read_ir):
        return


class SparkBackend(Backend):
    def _to_java_ir(self, ir):
        if not hasattr(ir, '_jir'):
            r = Renderer(stop_at_jir=True)
            code = r(ir)
            # FIXME parse should be static
            ir._jir = ir.parse(code, ir_map=r.jirs)
        return ir._jir

    def execute(self, ir):
        return ir.typ._from_json(
            Env.hail().expr.ir.Interpret.interpretJSON(
                self._to_java_ir(ir)))

    def table_read_type(self, tir):
        jir = self._to_java_ir(tir)
        return ttable._from_java(jir.typ())

    def matrix_read_type(self, mir):
        jir = self._to_java_ir(mir)
        return tmatrix._from_java(jir.typ())


class ServiceBackend(Backend):
    def __init__(self, host, port=80, scheme='http'):
        self.scheme = scheme
        self.host = host
        self.port = port

    def execute(self, ir):
        r = Renderer(stop_at_jir=True)
        code = r(ir)
        assert len(r.jirs) == 0
        
        resp = requests.post(f'http://hail-apiserver:5000/execute', json=code)
        resp.raise_for_status()
        
        resp_json = resp.json()
        
        typ = dtype(resp_json['type'])
        result = resp_json['value']
        
        return typ._from_json(result)
