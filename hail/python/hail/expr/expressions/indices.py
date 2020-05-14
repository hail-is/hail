from hail.typecheck import typecheck_method, anytype, setof
import hail as hl

from typing import List


class Indices(object):
    @typecheck_method(source=anytype, axes=setof(str))
    def __init__(self, source=None, axes=set()):
        self.source = source
        self.axes = axes
        self._cached_key = None

    def __hash__(self):
        return 37 + hash((self.source, *self.axes))

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

    @property
    def protected_key(self) -> List[str]:
        if self._cached_key is None:
            self._cached_key = self._get_key()
            return self._cached_key
        else:
            return self._cached_key

    def _get_key(self):
        if self.source is None:
            return []
        elif isinstance(self.source, hl.Table):
            if self == self.source._row_indices:
                return list(self.source.key)
            else:
                return []
        else:
            assert isinstance(self.source, hl.MatrixTable)
            if self == self.source._row_indices:
                return list(self.source.row_key)
            elif self == self.source._col_indices:
                return list(self.source.col_key)
            else:
                return []

    def __str__(self):
        return 'Indices(axes={}, source={})'.format(self.axes, self.source)

    def __repr__(self):
        return 'Indices(axes={}, source={})'.format(repr(self.axes), repr(self.source))


class Aggregation(object):
    def __init__(self, *exprs):
        self.exprs = exprs
        from ..expressions import unify_all
        indices, agg = unify_all(*exprs)
        self.nested = agg
        self.indices = indices

    def agg_axes(self):
        s = self.indices.axes.copy()
        for a in self.nested:
            s = s.union(a.agg_axes())
        return s
