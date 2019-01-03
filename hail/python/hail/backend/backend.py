import abc

from hail.utils.java import *
from hail.expr.types import dtype
from hail.expr.table_type import *
from hail.expr.matrix_type import *
from hail.ir.renderer import Renderer
from hail.table import Table

import pyspark

class Backend(object):
    @abc.abstractmethod
    def execute(self, ir):
        return

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

    def from_spark(self, df, key):
        return Table._from_java(Env.hail().table.Table.fromDF(Env.hc()._jhc, df._jdf, key))

    def to_spark(self, t, flatten):
        t = t.expand_types()
        if flatten:
            t = t.flatten()
        return pyspark.sql.DataFrame(t._jt.toDF(Env.hc()._jsql_context), Env.sql_context())

    def to_pandas(self, t, flatten):
        return self.to_spark(t, flatten).toPandas()

    def from_pandas(self, df, key):
        return Table.from_spark(Env.sql_context().createDataFrame(df), key)

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
