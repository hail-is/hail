import abc
from .renderer import Renderer

class BaseIR(object):
    def __init__(self):
        super().__init__()

    def __str__(self):
        r = Renderer(stop_at_jir = False)
        return r(self)

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
        others = [node for child in self.children for node in child.search(criteria)]
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


class TableIR(BaseIR):
    def __init__(self):
        super().__init__()


class MatrixIR(BaseIR):
    def __init__(self):
        super().__init__()
