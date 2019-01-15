import abc
from .renderer import Renderer
from hail.expr.matrix_type import *
from hail.expr.table_type import *
from hail.utils.java import Env


class BaseIR(object):
    def __init__(self):
        super().__init__()
        self._type = None

    def __str__(self):
        r = Renderer(stop_at_jir = False)
        return r(self)

    @abc.abstractmethod
    def parse(self, code, ref_map, ir_map):
        return

    @abc.abstractproperty
    def typ(self):
        return


class IR(BaseIR):
    def __init__(self, *children):
        super().__init__()
        self._aggregations = None
        self.children = children

    @property
    def aggregations(self):
        if self._aggregations is None:
            self._aggregations = [agg for child in self.children for agg in child.aggregations]
        return self._aggregations

    @property
    def is_nested_field(self):
        return False

    def search(self, criteria):
        others = [node for child in self.children if isinstance(child, IR) for node in child.search(criteria)]
        if criteria(self):
            return others + [self]
        return others

    def copy(self, *args):
        raise NotImplementedError("IR has no copy method defined.")

    def map_ir(self, f):
        new_children = []
        for child in self.children:
            if isinstance(child, IR):
                new_children.append(f(child))
            else:
                new_children.append(child)

        return self.copy(*new_children)

    @property
    def bound_variables(self):
        return {v for child in self.children for v in child.bound_variables}

    @property
    def typ(self):
        if self._type is None:
            self._compute_type({}, None)
            assert self._type is not None, self
        return self._type

    @abc.abstractmethod
    def _compute_type(self, env, agg_env):
        raise NotImplementedError(self)

    def parse(self, code, ref_map={}, ir_map={}):
        return Env.hail().expr.ir.IRParser.parse_value_ir(code, ref_map, ir_map)


class TableIR(BaseIR):
    def __init__(self):
        super().__init__()

    @abc.abstractmethod
    def _compute_type(self):
        raise NotImplementedError(self)

    @property
    def typ(self):
        if self._type is None:
            self._compute_type()
            assert self._type is not None, self
        return self._type

    def parse(self, code, ref_map={}, ir_map={}):
        return Env.hail().expr.ir.IRParser.parse_table_ir(code, ref_map, ir_map)


class MatrixIR(BaseIR):
    def __init__(self):
        super().__init__()

    @abc.abstractmethod
    def _compute_type(self):
        raise NotImplementedError(self)

    @property
    def typ(self):
        if self._type is None:
            self._compute_type()
            assert self._type is not None, self
        return self._type

    def parse(self, code, ref_map={}, ir_map={}):
        return Env.hail().expr.ir.IRParser.parse_matrix_ir(code, ref_map, ir_map)
