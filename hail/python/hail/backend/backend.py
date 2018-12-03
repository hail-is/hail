import abc

from hail.utils.java import *
from hail.expr.types import dtype
from hail.ir.renderer import Renderer


class Backend(object):
    @abc.abstractmethod
    def interpret(self, ir):
        return


class SparkBackend(Backend):
    def interpret(self, ir):
        assert isinstance(ir, hail.ir.IR)

        r = Renderer(stop_at_jir=True)
        code = r(ir)
        ir_map = {name: jir for name, jir in r.jirs.items()}

        jir = ir.to_java_ir()

        typ = dtype(jir.typ().toString())
        result = Env.hail().expr.ir.Interpret.interpretPyIR(code, {}, ir_map)

        return typ._from_json(result)


class ServiceBackend(Backend):
    def __init__(self, host, port=80, scheme='http'):
        self.scheme = scheme
        self.host = host
        self.port = port

    def interpret(self, ir):
        assert isinstance(ir, hail.ir.IR)

        r = Renderer(stop_at_jir=True)
        code = r(ir)
        assert len(r.jirs) == 0
        
        resp = requests.post(f'http://hail-apiserver:5000/execute', json=code)
        resp.raise_for_status()
        
        resp_json = resp.json()
        
        typ = dtype(resp_json['type'])
        result = resp_json['value']
        
        return typ._from_json(result)
