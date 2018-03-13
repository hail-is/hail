from hail.typecheck import *

class Indices(object):
    @typecheck_method(source=anytype, axes=setof(str))
    def __init__(self, source=None, axes=set()):
        self.source = source
        self.axes = axes

    def __eq__(self, other):
        return isinstance(other, Indices) and self.source is other.source and self.axes == other.axes

    def __ne__(self, other):
        return not self.__eq__(other)

    @staticmethod
    def unify(*indices):
        axes = set()
        src = None
        for ind in indices:
            if src is None:
                src = ind.source
            else:
                if ind.source is not None and ind.source is not src:
                    from . import ExpressionException
                    raise ExpressionException()

            axes = axes.union(ind.axes)

        return Indices(src, axes)

    def __str__(self):
        return 'Indices(axes={}, source={})'.format(self.axes, self.source)

    def __repr__(self):
        return 'Indices(axes={}, source={})'.format(repr(self.axes), repr(self.source))


class Aggregation(object):
    def __init__(self, *exprs):
        self.exprs = exprs
        from ..expressions import unify_all
        indices, agg, _ = unify_all(*exprs)
        self.indices = indices


class Join(object):
    def __init__(self, join_function, temp_vars, uid, exprs):
        self.join_function = join_function
        self.temp_vars = temp_vars
        self.uid = uid
        self.exprs = exprs
