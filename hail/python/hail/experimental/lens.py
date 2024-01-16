import abc

import hail as hl


class TableLike(abc.ABC):
    @abc.abstractmethod
    def annotate(self, *args, **kwargs):
        raise NotImplementedError

    @abc.abstractmethod
    def drop(self, *args, **kwargs):
        raise NotImplementedError

    @abc.abstractmethod
    def select(self, *args, **kwargs):
        raise NotImplementedError

    @abc.abstractmethod
    def explode(self, *args, **kwargs):
        raise NotImplementedError

    @abc.abstractmethod
    def group_by(self, *args, **kwargs):
        raise NotImplementedError

    @abc.abstractmethod
    def __getitem__(self, *args, **kwargs):
        raise NotImplementedError

    @abc.abstractmethod
    def index(self, *args, **kwargs):
        raise NotImplementedError

    @abc.abstractmethod
    def unlens(self):
        raise NotImplementedError


class GroupedTableLike(abc.ABC):
    @abc.abstractmethod
    def aggregate(self, *args, **kwargs):
        raise NotImplementedError


class MatrixRows(TableLike):
    def __init__(self, mt):
        assert isinstance(mt, hl.MatrixTable)
        self.mt = mt
        self.key = mt.row_key

    def annotate(self, *args, **kwargs):
        return MatrixRows(self.mt.annotate_rows(*args, **kwargs))

    def drop(self, *args, **kwargs):
        return MatrixRows(self.mt.drop(*args, **kwargs))

    def select(self, *args, **kwargs):
        return MatrixRows(self.mt.select_rows(*args, **kwargs))

    def explode(self, *args, **kwargs):
        return MatrixRows(self.mt.explode_rows(*args, **kwargs))

    def group_by(self, *args, **kwargs):
        return GroupedMatrixRows(self.mt.group_rows_by(*args, **kwargs))

    def __getitem__(self, *args, **kwargs):
        return self.mt.__getitem__(*args, **kwargs)

    def index(self, *args, **kwargs):
        return self.mt.rows().index(*args, **kwargs)

    def unlens(self):
        return self.mt


class GroupedMatrixRows(GroupedTableLike):
    def __init__(self, mt):
        assert isinstance(mt, hl.GroupedMatrixTable)
        self.mt = mt

    def aggregate(self, *args, **kwargs):
        return MatrixRows(self.mt.aggregate_rows(*args, **kwargs).result())


class TableRows(TableLike):
    def __init__(self, t):
        assert isinstance(t, hl.Table)
        self.t = t
        self.key = t.key

    def annotate(self, *args, **kwargs):
        return TableRows(self.t.annotate(*args, **kwargs))

    def drop(self, *args, **kwargs):
        return TableRows(self.t.drop(*args, **kwargs))

    def select(self, *args, **kwargs):
        return TableRows(self.t.select(*args, **kwargs))

    def explode(self, *args, **kwargs):
        return TableRows(self.t.explode(*args, **kwargs))

    def group_by(self, *args, **kwargs):
        return GroupedTableRows(self.t.group_by(*args, **kwargs))

    def __getitem__(self, *args, **kwargs):
        return self.t.__getitem__(*args, **kwargs)

    def index(self, *args, **kwargs):
        return self.t.index(*args, **kwargs)

    def unlens(self):
        return self.t


class GroupedTableRows(GroupedTableLike):
    def __init__(self, t):
        assert isinstance(t, hl.GroupedTable)
        self.t = t

    def aggregate(self, *args, **kwargs):
        return TableRows(self.t.aggregate(*args, **kwargs))
