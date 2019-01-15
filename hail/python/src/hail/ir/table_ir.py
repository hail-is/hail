import json

import hail as hl
from hail.ir.base_ir import *
from hail.utils.java import Env, escape_str, escape_id, parsable_strings, dump_json


class MatrixRowsTable(TableIR):
    def __init__(self, child):
        super().__init__()
        self.child = child

    def render(self, r):
        return '(MatrixRowsTable {})'.format(r(self.child))

    def _compute_type(self):
        self._type = hl.ttable(self.child.typ.global_type,
                               self.child.typ.row_type,
                               self.child.typ.row_key)

class TableJoin(TableIR):
    def __init__(self, left, right, join_type, join_key):
        super().__init__()
        self.left = left
        self.right = right
        self.join_type = join_type
        self.join_key = join_key

    def render(self, r):
        return '(TableJoin {} {} {} {})'.format(
            escape_id(self.join_type), self.join_key, r(self.left), r(self.right))

    def _compute_type(self):
        left_typ = self.left.typ
        right_typ = self.right.typ
        self._type = hl.ttable(left_typ.global_type._concat(right_typ.global_type),
                               left_typ.key_type._concat(left_typ.value_type)._concat(right_typ.value_type),
                               left_typ.row_key + right_typ.row_key[self.join_key:])

class TableLeftJoinRightDistinct(TableIR):
    def __init__(self, left, right, root):
        super().__init__()
        self.left = left
        self.right = right
        self.root = root

    def render(self, r):
        return '(TableLeftJoinRightDistinct {} {} {})'.format(
            escape_id(self.root), r(self.left), r(self.right))

    def _compute_type(self):
        left_typ = self.left.typ
        right_typ = self.right.typ
        self._type = hl.ttable(
            left_typ.global_type,
            left_typ.row_type._insert_field(self.root, right_typ.value_type),
            left_typ.row_key)

class TableIntervalJoin(TableIR):
    def __init__(self, left, right, root):
        super().__init__()
        self.left = left
        self.right = right
        self.root = root

    def render(self, r):
        return '(TableIntervalJoin {} {} {})'.format(
            escape_id(self.root), r(self.left), r(self.right))

    def _compute_type(self):
        left_typ = self.left.typ
        right_typ = self.right.typ
        self._type = hl.ttable(
            left_typ.global_type,
            left_typ.row_type._insert_field(self.root, right_typ.value_type),
            left_typ.row_key)


class TableUnion(TableIR):
    def __init__(self, children):
        super().__init__()
        self.children = children

    def render(self, r):
        return '(TableUnion {})'.format(' '.join([r(x) for x in self.children]))

    def _compute_type(self):
        for c in self.children:
            c.typ # force
        self._type = self.children[0].typ


class TableRange(TableIR):
    def __init__(self, n, n_partitions):
        super().__init__()
        self.n = n
        self.n_partitions = n_partitions

    def render(self, r):
        return '(TableRange {} {})'.format(self.n, self.n_partitions)

    def _compute_type(self):
        self._type = hl.ttable(hl.tstruct(),
                               hl.tstruct(idx=hl.tint32),
                               ['idx'])

class TableMapGlobals(TableIR):
    def __init__(self, child, new_row):
        super().__init__()
        self.child = child
        self.new_row = new_row

    def render(self, r):
        return '(TableMapGlobals {} {})'.format(r(self.child), r(self.new_row))

    def _compute_type(self):
        self.new_row._compute_type(self.child.typ.global_env(), None)
        self._type = hl.ttable(self.new_row.typ,
                               self.child.typ.row_type,
                               self.child.typ.row_key)

class TableExplode(TableIR):
    def __init__(self, child, path):
        super().__init__()
        self.child = child
        self.path = path

    def render(self, r):
        return '(TableExplode {} {})'.format(parsable_strings(self.path), r(self.child))

    def _compute_type(self):
        atyp = self.child.typ.row_type._index_path(self.path)
        self._type = hl.ttable(self.child.typ.global_type,
                               self.child.typ.row_type._insert(self.path, atyp.element_type),
                               self.child.typ.row_key)


class TableKeyBy(TableIR):
    def __init__(self, child, keys, is_sorted=False):
        super().__init__()
        self.child = child
        self.keys = keys
        self.is_sorted = is_sorted

    def render(self, r):
        return '(TableKeyBy ({}) {} {})'.format(
            ' '.join([escape_id(x) for x in self.keys]),
            self.is_sorted,
            r(self.child))

    def _compute_type(self):
        self._type = hl.ttable(self.child.typ.global_type,
                               self.child.typ.row_type,
                               self.keys)


class TableMapRows(TableIR):
    def __init__(self, child, new_row):
        super().__init__()
        self.child = child
        self.new_row = new_row

    def render(self, r):
        return '(TableMapRows {} {})'.format(r(self.child), r(self.new_row))

    def _compute_type(self):
        # agg_env for scans
        self.new_row._compute_type(self.child.typ.row_env(), self.child.typ.row_env())
        self._type = hl.ttable(
            self.child.typ.global_type,
            self.new_row.typ,
            self.child.typ.row_key)

class TableRead(TableIR):
    def __init__(self, path, drop_rows, typ):
        super().__init__()
        self.path = path
        self.drop_rows = drop_rows
        self._typ = typ

    def render(self, r):
        return '(TableRead "{}" {} {})'.format(
            escape_str(self.path),
            self.drop_rows,
            self._typ)

    def _compute_type(self):
        self._type = Env.backend().table_type(self)


class TableImport(TableIR):
    def __init__(self, paths, typ, reader_options):
        super().__init__()
        self.paths = paths
        self._typ = typ
        self.reader_options = reader_options

    def render(self, r):
        return '(TableImport ({}) {} {})'.format(
            ' '.join([escape_str(path) for path in self.paths]),
            self._typ._parsable_string(),
            escape_str(json.dumps(self.reader_options)))

    def _compute_type(self):
        self._type = Env.backend().table_type(self)


class MatrixEntriesTable(TableIR):
    def __init__(self, child):
        super().__init__()
        self.child = child

    def render(self, r):
        return '(MatrixEntriesTable {})'.format(r(self.child))

    def _compute_type(self):
        child_typ = self.child.typ
        self._type = hl.ttable(child_typ.global_type,
                               child_typ.row_type
                               ._concat(child_typ.col_type)
                               ._concat(child_typ.entry_type),
                               child_typ.row_key + child_typ.col_key)


class TableFilter(TableIR):
    def __init__(self, child, pred):
        super().__init__()
        self.child = child
        self.pred = pred

    def render(self, r):
        return '(TableFilter {} {})'.format(r(self.child), r(self.pred))

    def _compute_type(self):
        self.pred._compute_type(self.child.typ.row_env(), None)
        self._type = self.child.typ


class TableKeyByAndAggregate(TableIR):
    def __init__(self, child, expr, new_key, n_partitions, buffer_size):
        super().__init__()
        self.child = child
        self.expr = expr
        self.new_key = new_key
        self.n_partitions = n_partitions
        self.buffer_size = buffer_size

    def render(self, r):
        return '(TableKeyByAndAggregate {} {} {} {} {})'.format(self.n_partitions,
                                                                self.buffer_size,
                                                                r(self.child),
                                                                r(self.expr),
                                                                self.new_key)

    def _compute_type(self):
        self.expr._compute_type(self.child.typ.global_env(), self.child.typ.row_env())
        self.new_key._compute_type(self.child.typ.row_env(), None)
        self._type = hl.ttable(self.child.typ.global_type,
                               self.new_key.typ._concat(self.expr.typ),
                               list(self.new_key.typ))


class TableAggregateByKey(TableIR):
    def __init__(self, child, expr):
        super().__init__()
        self.child = child
        self.expr = expr

    def render(self, r):
        return '(TableAggregateByKey {} {})'.format(r(self.child), r(self.expr))

    def _compute_type(self):
        child_typ = self.child.typ
        self.expr._compute_type(child_typ.global_env(), child_typ.row_env())
        self._type = hl.ttable(child_typ.global_type,
                               child_typ.key_type._concat(self.expr.typ),
                               child_typ.row_key)

class MatrixColsTable(TableIR):
    def __init__(self, child):
        super().__init__()
        self.child = child

    def render(self, r):
        return '(MatrixColsTable {})'.format(r(self.child))

    def _compute_type(self):
        self._type = hl.ttable(self.child.typ.global_type,
                               self.child.typ.col_type,
                               self.child.typ.col_key)


class TableParallelize(TableIR):
    def __init__(self, rows_and_global, n_partitions):
        super().__init__()
        self.rows_and_global = rows_and_global
        self.n_partitions = n_partitions

    def render(self, r):
        return '(TableParallelize {} {})'.format(
            self.n_partitions,
            r(self.rows_and_global))

    def _compute_type(self):
        self.rows_and_global._compute_type({}, None)
        self._type = hl.ttable(self.rows_and_global.typ['global'],
                               self.rows_and_global.typ['rows'].element_type,
                               [])


class TableHead(TableIR):
    def __init__(self, child, n):
        super().__init__()
        self.child = child
        self.n = n

    def render(self, r):
        return f'(TableHead {self.n} {r(self.child)})'

    def _compute_type(self):
        self._type = self.child.typ


class TableOrderBy(TableIR):
    def __init__(self, child, sort_fields):
        super().__init__()
        self.child = child
        self.sort_fields = sort_fields

    def render(self, r):
        return '(TableOrderBy ({}) {})'.format(
            ' '.join(['{}{}'.format(order, escape_id(f)) for (f, order) in self.sort_fields]),
            r(self.child))

    def _compute_type(self):
        self._type = hl.ttable(self.child.typ.global_type,
                               self.child.typ.row_type,
                               [])


class TableDistinct(TableIR):
    def __init__(self, child):
        super().__init__()
        self.child = child

    def render(self, r):
        return f'(TableDistinct {r(self.child)})'

    def _compute_type(self):
        self._type = self.child.typ

class RepartitionStrategy:
    SHUFFLE = 0
    COALESCE = 1
    NAIVE_COALESCE = 2

class TableRepartition(TableIR):
    def __init__(self, child, n, strategy):
        super().__init__()
        self.child = child
        self.n = n
        self.strategy = strategy

    def render(self, r):
        return f'(TableRepartition {self.n} {self.strategy} {r(self.child)})'

    def _compute_type(self):
        self._type = self.child.typ


class CastMatrixToTable(TableIR):
    def __init__(self, child, entries_field_name, cols_field_name):
        super().__init__()
        self.child = child
        self.entries_field_name = entries_field_name
        self.cols_field_name = cols_field_name

    def render(self, r):
        return f'(CastMatrixToTable ' \
               f'"{escape_str(self.entries_field_name)}" ' \
               f'"{escape_str(self.cols_field_name)}" ' \
               f'{r(self.child)})'

    def _compute_type(self):
        child_typ = self.child.typ
        self._type = hl.ttable(child_typ.global_type._insert_field(self.cols_field_name, hl.tarray(child_typ.col_type)),
                               child_typ.row_type._insert_field(self.entries_field_name, hl.tarray(child_typ.entry_type)),
                               child_typ.row_key)


class TableRename(TableIR):
    def __init__(self, child, row_map, global_map):
        super().__init__()
        self.child = child
        self.row_map = row_map
        self.global_map = global_map

    def render(self, r):
        return f'(TableRename ' \
               f'{parsable_strings(self.row_map.keys())} ' \
               f'{parsable_strings(self.row_map.values())} ' \
               f'{parsable_strings(self.global_map.keys())} ' \
               f'{parsable_strings(self.global_map.values())} ' \
               f'{r(self.child)})'

    def _compute_type(self):
        self._type = self.child.typ._rename(self.global_map, self.row_map)


class TableMultiWayZipJoin(TableIR):
    def __init__(self, children, data_name, global_name):
        super().__init__()
        self.children = children
        self.data_name = data_name
        self.global_name = global_name

    def render(self, r):
        return f'(TableMultiWayZipJoin '\
               f'"{escape_str(self.data_name)}" '\
               f'"{escape_str(self.global_name)}" '\
               f'{" ".join([r(child) for child in self.children])})'

    def _compute_type(self):
        for c in self.children:
            c.typ # force
        child_typ = self.children[0].typ
        self._type = hl.ttable(
            hl.tstruct(**{self.global_name: hl.tarray(child_typ.global_type)}),
            child_typ.key_type._insert_field(self.data_name, hl.tarray(child_typ.value_type)),
            child_typ.row_key)


class TableToTableApply(TableIR):
    def __init__(self, child, config):
        super().__init__()
        self.child = child
        self.config = config

    def render(self, r):
        return f'(TableToTableApply {dump_json(self.config)} {r(self.child)})'

    def _compute_type(self):
        assert self.config['name'] == 'TableFilterPartitions'
        self._type = self.child.typ


class MatrixToTableApply(TableIR):
    def __init__(self, child, config):
        super().__init__()
        self.child = child
        self.config = config

    def render(self, r):
        return f'(MatrixToTableApply {dump_json(self.config)} {r(self.child)})'

    def _compute_type(self):
        name = self.config['name']
        child_typ = self.child.typ
        pass_through = self.config['passThrough']
        if name == 'LinearRegressionRowsChained':
            chained_schema = hl.dtype('struct{n:array<int32>,sum_x:array<float64>,y_transpose_x:array<array<float64>>,beta:array<array<float64>>,standard_error:array<array<float64>>,t_stat:array<array<float64>>,p_value:array<array<float64>>}')
            self._type = hl.ttable(
                child_typ.global_type,
                (child_typ.row_key_type
                 ._insert_fields(**{f: child_typ.row_type[f] for f in pass_through})
                 ._concat(chained_schema)),
                child_typ.row_key)
        else:
            assert name == 'LinearRegressionRowsSingle', name
            chained_schema = hl.dtype('struct{n:int32,sum_x:float64,y_transpose_x:array<float64>,beta:array<float64>,standard_error:array<float64>,t_stat:array<float64>,p_value:array<float64>}')
            self._type = hl.ttable(
                child_typ.global_type,
                (child_typ.row_key_type
                 ._insert_fields(**{f: child_typ.row_type[f] for f in pass_through})
                 ._concat(chained_schema)),
                child_typ.row_key)


class JavaTable(TableIR):
    def __init__(self, jir):
        super().__init__()
        self._jir = jir

    def render(self, r):
        return f'(JavaTable {r.add_jir(self._jir)})'

    def _compute_type(self):
        self._type = hl.ttable._from_java(self._jir.typ())
