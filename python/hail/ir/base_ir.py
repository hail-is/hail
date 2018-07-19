import abc

class BaseIR(object):
    def __init__(self):
        super().__init__()


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

    @property
    def bound_variables(self):
        return {v for child in self.children for v in child.bound_variables}


class TableIR(BaseIR):
    def __init__(self):
        super().__init__()


class MatrixIR(BaseIR):
    def __init__(self):
        super().__init__()
