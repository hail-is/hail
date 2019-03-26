import json

import hail as hl
from hail.expr.types import dtype
from hail.ir.base_ir import *
from hail.utils.java import Env, escape_str, escape_id, parsable_strings, dump_json


class MatrixRowsTable(TableIR):
    def __init__(self, child):
        super().__init__(child)
        self.child = child

    def _compute_type(self):
        self._type = hl.ttable(self.child.typ.global_type,
                               self.child.typ.row_type,
                               self.child.typ.row_key)


class TableJoin(TableIR):
    def __init__(self, left, right, join_type, join_key):
        super().__init__(left, right)
        self.left = left
        self.right = right
        self.join_type = join_type
        self.join_key = join_key

    def head_str(self):
        return f'{escape_id(self.join_type)} {self.join_key}'

    def _eq(self, other):
        return self.join_key == other.join_key and \
               self.join_type == other.join_type

    def _compute_type(self):
        left_typ = self.left.typ
        right_typ = self.right.typ
        self._type = hl.ttable(left_typ.global_type._concat(right_typ.global_type),
                               left_typ.key_type._concat(left_typ.value_type)._concat(right_typ.value_type),
                               left_typ.row_key + right_typ.row_key[self.join_key:])


class TableLeftJoinRightDistinct(TableIR):
    def __init__(self, left, right, root):
        super().__init__(left, right)
        self.left = left
        self.right = right
        self.root = root

    def head_str(self):
        return escape_id(self.root)

    def _eq(self, other):
        return self.root == other.root

    def _compute_type(self):
        left_typ = self.left.typ
        right_typ = self.right.typ
        self._type = hl.ttable(
            left_typ.global_type,
            left_typ.row_type._insert_field(self.root, right_typ.value_type),
            left_typ.row_key)


class TableIntervalJoin(TableIR):
    def __init__(self, left, right, root):
        super().__init__(left, right)
        self.left = left
        self.right = right
        self.root = root

    def head_str(self):
        return escape_id(self.root)

    def _eq(self, other):
        return self.root == other.root

    def _compute_type(self):
        left_typ = self.left.typ
        right_typ = self.right.typ
        self._type = hl.ttable(
            left_typ.global_type,
            left_typ.row_type._insert_field(self.root, right_typ.value_type),
            left_typ.row_key)


class TableUnion(TableIR):
    def __init__(self, children):
        super().__init__(*children)
        self.children = children

    def _compute_type(self):
        for c in self.children:
            c.typ  # force
        self._type = self.children[0].typ


class TableRange(TableIR):
    def __init__(self, n, n_partitions):
        super().__init__()
        self.n = n
        self.n_partitions = n_partitions

    def head_str(self):
        return f'{self.n} {self.n_partitions}'

    def _eq(self, other):
        return self.n == other.n and self.n_partitions == other.n_partitions

    def _compute_type(self):
        self._type = hl.ttable(hl.tstruct(),
                               hl.tstruct(idx=hl.tint32),
                               ['idx'])


class TableMapGlobals(TableIR):
    def __init__(self, child, new_globals):
        super().__init__(child, new_globals)
        self.child = child
        self.new_globals = new_globals

    def _compute_type(self):
        self.new_globals._compute_type(self.child.typ.global_env(), None)
        self._type = hl.ttable(self.new_globals.typ,
                               self.child.typ.row_type,
                               self.child.typ.row_key)


class TableExplode(TableIR):
    def __init__(self, child, path):
        super().__init__(child)
        self.child = child
        self.path = path

    def head_str(self):
        return parsable_strings(self.path)

    def _eq(self, other):
        return self.path == other.path

    def _compute_type(self):
        atyp = self.child.typ.row_type._index_path(self.path)
        self._type = hl.ttable(self.child.typ.global_type,
                               self.child.typ.row_type._insert(self.path, atyp.element_type),
                               self.child.typ.row_key)


class TableKeyBy(TableIR):
    def __init__(self, child, keys, is_sorted=False):
        super().__init__(child)
        self.child = child
        self.keys = keys
        self.is_sorted = is_sorted

    def head_str(self):
        return '({}) {}'.format(' '.join([escape_id(x) for x in self.keys]), self.is_sorted)

    def _eq(self, other):
        return self.keys == other.keys and self.is_sorter == other.is_sorted

    def _compute_type(self):
        self._type = hl.ttable(self.child.typ.global_type,
                               self.child.typ.row_type,
                               self.keys)


class TableMapRows(TableIR):
    def __init__(self, child, new_row):
        super().__init__(child, new_row)
        self.child = child
        self.new_row = new_row

    def _compute_type(self):
        # agg_env for scans
        self.new_row._compute_type(self.child.typ.row_env(), self.child.typ.row_env())
        self._type = hl.ttable(
            self.child.typ.global_type,
            self.new_row.typ,
            self.child.typ.row_key)


class TableRead(TableIR):
    def __init__(self, reader, drop_rows=False):
        super().__init__()
        self.reader = reader
        self.drop_rows = drop_rows

    def head_str(self):
        return f'None {self.drop_rows} "{self.reader.render()}"'

    def _eq(self, other):
        return self.reader == other.reader and self.drop_rows == other.drop_rows

    def _compute_type(self):
        self._type = Env.backend().table_type(self)


class TableImport(TableIR):
    def __init__(self, paths, typ, reader_options):
        super().__init__()
        self.paths = paths
        self._typ = typ
        self.reader_options = reader_options

    def head_str(self):
        return '(({}) {} {}'.format(
            ' '.join([escape_str(path) for path in self.paths]),
            self._typ._parsable_string(),
            escape_str(json.dumps(self.reader_options)))

    def _eq(self, other):
        return self.paths == other.paths and self.typ == other.typ and self.reader_options == other.reader_options

    def _compute_type(self):
        self._type = Env.backend().table_type(self)


class MatrixEntriesTable(TableIR):
    def __init__(self, child):
        super().__init__(child)
        self.child = child

    def _compute_type(self):
        child_typ = self.child.typ
        self._type = hl.ttable(child_typ.global_type,
                               child_typ.row_type
                               ._concat(child_typ.col_type)
                               ._concat(child_typ.entry_type),
                               child_typ.row_key + child_typ.col_key)


class TableFilter(TableIR):
    def __init__(self, child, pred):
        super().__init__(child, pred)
        self.child = child
        self.pred = pred

    def _compute_type(self):
        self.pred._compute_type(self.child.typ.row_env(), None)
        self._type = self.child.typ


class TableKeyByAndAggregate(TableIR):
    def __init__(self, child, expr, new_key, n_partitions, buffer_size):
        super().__init__(child, expr, new_key)
        self.child = child
        self.expr = expr
        self.new_key = new_key
        self.n_partitions = n_partitions
        self.buffer_size = buffer_size

    def head_str(self):
        return f'{self.n_partitions} {self.buffer_size}'

    def _eq(self, other):
        return self.n_partitions == other.n_partitions and self.buffer_size == other.buffer_size

    def _compute_type(self):
        self.expr._compute_type(self.child.typ.global_env(), self.child.typ.row_env())
        self.new_key._compute_type(self.child.typ.row_env(), None)
        self._type = hl.ttable(self.child.typ.global_type,
                               self.new_key.typ._concat(self.expr.typ),
                               list(self.new_key.typ))


class TableAggregateByKey(TableIR):
    def __init__(self, child, expr):
        super().__init__(child, expr)
        self.child = child
        self.expr = expr

    def _compute_type(self):
        child_typ = self.child.typ
        self.expr._compute_type(child_typ.global_env(), child_typ.row_env())
        self._type = hl.ttable(child_typ.global_type,
                               child_typ.key_type._concat(self.expr.typ),
                               child_typ.row_key)


class MatrixColsTable(TableIR):
    def __init__(self, child):
        super().__init__(child)
        self.child = child

    def _compute_type(self):
        self._type = hl.ttable(self.child.typ.global_type,
                               self.child.typ.col_type,
                               self.child.typ.col_key)


class TableParallelize(TableIR):
    def __init__(self, rows_and_global, n_partitions):
        super().__init__(rows_and_global)
        self.rows_and_global = rows_and_global
        self.n_partitions = n_partitions

    def head_str(self):
        return self.n_partitions

    def _eq(self, other):
        return self.n_partitions == other.n_partitions

    def _compute_type(self):
        self.rows_and_global._compute_type({}, None)
        self._type = hl.ttable(self.rows_and_global.typ['global'],
                               self.rows_and_global.typ['rows'].element_type,
                               [])


class TableHead(TableIR):
    def __init__(self, child, n):
        super().__init__(child)
        self.child = child
        self.n = n

    def head_str(self):
        return self.n

    def _eq(self, other):
        return self.n == other.n

    def _compute_type(self):
        self._type = self.child.typ


class TableOrderBy(TableIR):
    def __init__(self, child, sort_fields):
        super().__init__(child)
        self.child = child
        self.sort_fields = sort_fields

    def head_str(self):
        return f'({" ".join([escape_id(order + f) for (f, order) in self.sort_fields])})'

    def _eq(self, other):
        return self.sort_fields == other.sort_fields

    def _compute_type(self):
        self._type = hl.ttable(self.child.typ.global_type,
                               self.child.typ.row_type,
                               [])


class TableDistinct(TableIR):
    def __init__(self, child):
        super().__init__(child)
        self.child = child

    def _compute_type(self):
        self._type = self.child.typ


class RepartitionStrategy:
    SHUFFLE = 0
    COALESCE = 1
    NAIVE_COALESCE = 2


class TableRepartition(TableIR):
    def __init__(self, child, n, strategy):
        super().__init__(child)
        self.child = child
        self.n = n
        self.strategy = strategy

    def head_str(self):
        return f'{self.n} {self.strategy}'

    def _eq(self, other):
        return self.n == other.n and self.strategy == other.strategy

    def _compute_type(self):
        self._type = self.child.typ


class CastMatrixToTable(TableIR):
    def __init__(self, child, entries_field_name, cols_field_name):
        super().__init__(child)
        self.child = child
        self.entries_field_name = entries_field_name
        self.cols_field_name = cols_field_name

    def head_str(self):
        return f'"{escape_str(self.entries_field_name)}" "{escape_str(self.cols_field_name)}"'

    def _eq(self, other):
        return self.entries_field_name == other.entries_field_name and self.cols_field_name == other.cols_field_name

    def _compute_type(self):
        child_typ = self.child.typ
        self._type = hl.ttable(child_typ.global_type._insert_field(self.cols_field_name, hl.tarray(child_typ.col_type)),
                               child_typ.row_type._insert_field(self.entries_field_name,
                                                                hl.tarray(child_typ.entry_type)),
                               child_typ.row_key)


class TableRename(TableIR):
    def __init__(self, child, row_map, global_map):
        super().__init__(child)
        self.child = child
        self.row_map = row_map
        self.global_map = global_map

    def head_str(self):
        return f'{parsable_strings(self.row_map.keys())} ' \
               f'{parsable_strings(self.row_map.values())} ' \
               f'{parsable_strings(self.global_map.keys())} ' \
               f'{parsable_strings(self.global_map.values())} '

    def _eq(self, other):
        return self.row_map == other.row_map and self.global_map == other.global_map

    def _compute_type(self):
        self._type = self.child.typ._rename(self.global_map, self.row_map)


class TableMultiWayZipJoin(TableIR):
    def __init__(self, children, data_name, global_name):
        super().__init__(*children)
        self.children = children
        self.data_name = data_name
        self.global_name = global_name

    def head_str(self):
        return f'"{escape_str(self.data_name)}" "{escape_str(self.global_name)}"'

    def _eq(self, other):
        return self.data_name == other.data_name and self.global_name == other.global_name

    def _compute_type(self):
        for c in self.children:
            c.typ  # force
        child_typ = self.children[0].typ
        self._type = hl.ttable(
            hl.tstruct(**{self.global_name: hl.tarray(child_typ.global_type)}),
            child_typ.key_type._insert_field(self.data_name, hl.tarray(child_typ.value_type)),
            child_typ.row_key)


class TableToTableApply(TableIR):
    def __init__(self, child, config):
        super().__init__(child)
        self.child = child
        self.config = config

    def head_str(self):
        return dump_json(self.config)

    def _eq(self, other):
        return self.config == other.config

    def _compute_type(self):
        name = self.config['name']
        if name == 'TableFilterPartitions' or name == 'TableFilterIntervals':
            self._type = self.child.typ
        else:
            assert name in ('VEP', 'Nirvana'), name
            self._type = Env.backend().table_type(self)


def regression_test_type(test):
    glm_fit_schema = dtype('struct{n_iterations:int32,converged:bool,exploded:bool}')
    if test == 'wald':
        return dtype(
            f'struct{{beta:float64,standard_error:float64,z_stat:float64,p_value:float64,fit:{glm_fit_schema}}}')
    elif test == 'lrt':
        return dtype(f'struct{{beta:float64,chi_sq_stat:float64,p_value:float64,fit:{glm_fit_schema}}}')
    elif test == 'score':
        return dtype('struct{chi_sq_stat:float64,p_value:float64}')
    else:
        assert test == 'firth', test
        return dtype(f'struct{{beta:float64,chi_sq_stat:float64,p_value:float64,fit:{glm_fit_schema}}}')


class MatrixToTableApply(TableIR):
    def __init__(self, child, config):
        super().__init__(child)
        self.child = child
        self.config = config

    def head_str(self):
        return dump_json(self.config)

    def _eq(self, other):
        return self.config == other.config

    def _compute_type(self):
        name = self.config['name']
        child_typ = self.child.typ
        if name == 'LinearRegressionRowsChained':
            pass_through = self.config['passThrough']
            chained_schema = hl.dtype(
                'struct{n:array<int32>,sum_x:array<float64>,y_transpose_x:array<array<float64>>,beta:array<array<float64>>,standard_error:array<array<float64>>,t_stat:array<array<float64>>,p_value:array<array<float64>>}')
            self._type = hl.ttable(
                child_typ.global_type,
                (child_typ.row_key_type
                 ._insert_fields(**{f: child_typ.row_type[f] for f in pass_through})
                 ._concat(chained_schema)),
                child_typ.row_key)
        elif name == 'LinearRegressionRowsSingle':
            pass_through = self.config['passThrough']
            chained_schema = hl.dtype(
                'struct{n:int32,sum_x:float64,y_transpose_x:array<float64>,beta:array<float64>,standard_error:array<float64>,t_stat:array<float64>,p_value:array<float64>}')
            self._type = hl.ttable(
                child_typ.global_type,
                (child_typ.row_key_type
                 ._insert_fields(**{f: child_typ.row_type[f] for f in pass_through})
                 ._concat(chained_schema)),
                child_typ.row_key)
        elif name == 'LogisticRegression':
            pass_through = self.config['passThrough']
            logreg_type = hl.tstruct(logistic_regression=hl.tarray(regression_test_type(self.config['test'])))
            self._type = hl.ttable(
                child_typ.global_type,
                (child_typ.row_key_type
                 ._insert_fields(**{f: child_typ.row_type[f] for f in pass_through})
                 ._concat(logreg_type)),
                child_typ.row_key)
        elif name == 'PoissonRegression':
            pass_through = self.config['passThrough']
            poisreg_type = regression_test_type(self.config['test'])
            self._type = hl.ttable(
                child_typ.global_type,
                (child_typ.row_key_type
                 ._insert_fields(**{f: child_typ.row_type[f] for f in pass_through})
                 ._concat(poisreg_type)),
                child_typ.row_key)
        elif name == 'Skat':
            key_field = self.config['keyField']
            key_type = child_typ.row_type[key_field]
            skat_type = hl.dtype(f'struct{{id:{key_type},size:int32,q_stat:float64,p_value:float64,fault:int32}}')
            self._type = hl.ttable(
                hl.tstruct(),
                skat_type,
                ['id'])
        elif name == 'PCA':
            self._type = hl.ttable(
                hl.tstruct(eigenvalues=hl.tarray(hl.tfloat64),
                           scores=hl.tarray(child_typ.col_key_type._insert_field('scores', hl.tarray(hl.tfloat64)))),
                child_typ.row_key_type._insert_field('loadings', dtype('array<float64>')),
                child_typ.row_key)
        else:
            assert name == 'LocalLDPrune', name
            self._type = hl.ttable(
                hl.tstruct(),
                child_typ.row_key_type._insert_fields(mean=hl.tfloat64, centered_length_rec=hl.tfloat64),
                list(child_typ.row_key))


class BlockMatrixToTable(TableIR):
    def __init__(self, child):
        super().__init__(child)
        self.child = child

    def _compute_type(self):
        self._type = hl.ttable(hl.tstruct(), hl.tstruct(**{'i': hl.tint64, 'j': hl.tint64, 'entry': hl.tfloat64}), [])


class JavaTable(TableIR):
    def __init__(self, jir):
        super().__init__()
        self._jir = jir

    def render_head(self, r):
        return f'(JavaTable {r.add_jir(self._jir)}'

    def _compute_type(self):
        self._type = hl.ttable._from_java(self._jir.typ())
