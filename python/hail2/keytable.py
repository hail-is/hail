from __future__ import print_function  # Python 2 and 3 print compatibility

from hail2.expr.column import Column, convert_column, to_expr
from hail.java import *
from hail.typ import Type, TArray, TStruct, TAggregable
from hail.representation import Struct
from hail.typecheck import *
from hail.utils import wrap_to_list
from pyspark.sql import DataFrame


class KeyTableTemplate(object):
    def __init__(self, hc, jkt):
        self.hc = hc
        self._jkt = jkt

        self._schema = None
        self._num_columns = None
        self._key = None
        self._column_names = None

    def __getitem__(self, item):
        if item in self.columns:
            return self.columns[item]
        else:
            raise "Could not find column `" + item + "' in key table."

    def __setattr__(self, key, value):
        self.__dict__[key] = value

    def __delattr__(self, item):
        pass

    def __repr__(self):
        return self._jkt.toString()

    @property
    @handle_py4j
    def schema(self):
        if self._schema is None:
            self._schema = Type._from_java(self._jkt.signature())
            assert (isinstance(self._schema, TStruct))
        return self._schema


class GroupedKeyTable(KeyTableTemplate):
    def __init__(self, hc, jkt, groups):
        super(GroupedKeyTable, self).__init__(hc, jkt)
        self._groups = groups

        for fd in self.schema.fields:
            self.__setattr__(fd.name, convert_column(Column(fd.name, TAggregable(fd.typ))))

    @handle_py4j
    def aggregate_by_key(self, num_partitions=None, **kwargs):
        agg_expr = [k + " = " + to_expr(v) for k, v in kwargs.items()]
        return KeyTable(self.hc, self._jkt.aggregate(self._groups, ", ".join(agg_expr), joption(num_partitions)))


class AggregatedKeyTable(KeyTableTemplate):
    def __init__(self, hc, jkt):
        super(AggregatedKeyTable, self).__init__(hc, jkt)

        for fd in self.schema.fields:
            self.__setattr__(fd.name, convert_column(Column(fd.name, TAggregable(fd.typ))))

    @handle_py4j
    def query_typed(self, exprs):
        if isinstance(exprs, list):
            exprs = [to_expr(e) for e in exprs]
            result_list = self._jkt.query(jarray(Env.jvm().java.lang.String, exprs))
            ptypes = [Type._from_java(x._2()) for x in result_list]
            annotations = [ptypes[i]._convert_to_py(result_list[i]._1()) for i in xrange(len(ptypes))]
            return annotations, ptypes

        else:
            result = self._jkt.query(to_expr(exprs))
            t = Type._from_java(result._2())
            return t._convert_to_py(result._1()), t

    @handle_py4j
    def query(self, exprs):
        r, t = self.query_typed(exprs)
        return r


class KeyTable(KeyTableTemplate):
    def __init__(self, hc, jkt):
        super(KeyTable, self).__init__(hc, jkt)
        for fd in self.schema.fields:
            self.__setattr__(fd.name, convert_column(Column(fd.name, fd.typ)))

    @property
    @handle_py4j
    def columns(self):
        if self._column_names is None:
            self._column_names = list(self._jkt.columns())
        return self._column_names

    @property
    @handle_py4j
    def num_columns(self):
        if self._num_columns is None:
            self._num_columns = self._jkt.nColumns()
        return self._num_columns

    @property
    @handle_py4j
    def key(self):
        if self._key is None:
            self._key = list(self._jkt.key())
        return self._key

    @handle_py4j
    def count(self):
        return self._jkt.count()

    @classmethod
    @handle_py4j
    def parallelize(cls, rows, schema, key=[], num_partitions=None):
        return KeyTable(
            Env.hc(),
            Env.hail().keytable.KeyTable.parallelize(
                Env.hc()._jhc, [schema._convert_to_j(r) for r in rows],
                schema._jtype, wrap_to_list(key), joption(num_partitions)))

    @handle_py4j
    def annotate(self, **kwargs):
        exprs = [k + " = " + to_expr(v) for k, v in kwargs.items()]
        return KeyTable(self.hc, self._jkt.annotate(", ".join(exprs)))

    @handle_py4j
    def filter(self, expr, keep=True):
        jkt = self._jkt.filter(expr.expr, keep)
        return KeyTable(self.hc, jkt)

    @handle_py4j
    def select(self, *column_names):
        return KeyTable(self.hc, self._jkt.select(column_names))

    @handle_py4j
    def export(self, output, types_file=None, header=True):
        self._jkt.export(output, types_file, header)

    def aggregate(self):
        return AggregatedKeyTable(self.hc, self._jkt)

    def group_by(self, **kwargs):
        group_exprs = [k + " = " + to_expr(v) for k, v in kwargs.items()]
        return GroupedKeyTable(self.hc, self._jkt, ", ".join(group_exprs))

    def to_hail1(self):
        import hail
        return hail.KeyTable(self.hc, self._jkt)
