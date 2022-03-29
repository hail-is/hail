import hail as hl
from hail.ir.base_ir import BaseIR, MatrixIR
from hail.utils.misc import escape_str, parsable_strings, dump_json, escape_id
from hail.utils.java import Env


class MatrixAggregateRowsByKey(MatrixIR):
    def __init__(self, child, entry_expr, row_expr):
        super().__init__(child, entry_expr, row_expr)
        self.child = child
        self.entry_expr = entry_expr
        self.row_expr = row_expr

    def _compute_type(self):
        child_typ = self.child.typ
        self.entry_expr._compute_type(child_typ.col_env(), child_typ.entry_env())
        self.row_expr._compute_type(child_typ.global_env(), child_typ.row_env())
        self._type = hl.tmatrix(
            child_typ.global_type,
            child_typ.col_type,
            child_typ.col_key,
            child_typ.row_key_type._concat(self.row_expr.typ),
            child_typ.row_key,
            self.entry_expr.typ)

    def renderable_bindings(self, i, default_value=None):
        if i == 1:
            env = self.child.typ.col_env(default_value)
            env[BaseIR.agg_capability] = default_value
            return env
        elif i == 2:
            env = self.child.typ.global_env(default_value)
            env[BaseIR.agg_capability] = default_value
            return env
        else:
            return {}

    def renderable_agg_bindings(self, i, default_value=None):
        if i == 1:
            return self.child.typ.entry_env(default_value)
        elif i == 2:
            return self.child.typ.row_env(default_value)
        else:
            return {}


class MatrixRead(MatrixIR):
    def __init__(self, reader, drop_cols=False, drop_rows=False):
        super().__init__()
        self.reader = reader
        self.drop_cols = drop_cols
        self.drop_rows = drop_rows

    def render_head(self, r):
        return f'(MatrixRead None {self.drop_cols} {self.drop_rows} "{self.reader.render(r)}"'

    def _eq(self, other):
        return self.reader == other.reader and self.drop_cols == other.drop_cols and self.drop_rows == other.drop_rows

    def _compute_type(self):
        self._type = Env.backend().matrix_type(self)


class MatrixFilterRows(MatrixIR):
    def __init__(self, child, pred):
        super().__init__(child, pred)
        self.child = child
        self.pred = pred

    def _compute_type(self):
        self.pred._compute_type(self.child.typ.row_env(), None)
        self._type = self.child.typ

    def renderable_bindings(self, i, default_value=None):
        return self.child.typ.row_env(default_value) if i == 1 else {}


class MatrixChooseCols(MatrixIR):
    def __init__(self, child, old_indices):
        super().__init__(child)
        self.child = child
        self.old_indices = old_indices

    def head_str(self):
        return f'({" ".join([str(i) for i in self.old_indices])})'

    def _eq(self, other):
        return self.old_indices == other.old_indices

    def _compute_type(self):
        self._type = self.child.typ


class MatrixMapCols(MatrixIR):
    def __init__(self, child, new_col, new_key):
        super().__init__(child, new_col)
        self.child = child
        self.new_col = new_col
        self.new_key = new_key

    def head_str(self):
        return '(' + ' '.join(f'"{escape_str(f)}"' for f in self.new_key) + ')' if self.new_key is not None else 'None'

    def _eq(self, other):
        return self.new_key == other.new_key

    def _compute_type(self):
        child_typ = self.child.typ
        self.new_col._compute_type({**child_typ.col_env(), 'n_rows': hl.tint64}, child_typ.entry_env())
        self._type = hl.tmatrix(
            child_typ.global_type,
            self.new_col.typ,
            self.new_key if self.new_key is not None else child_typ.col_key,
            child_typ.row_type,
            child_typ.row_key,
            child_typ.entry_type)

    def renderable_bindings(self, i, default_value=None):
        if i == 1:
            env = self.child.typ.col_env(default_value)
            env[BaseIR.agg_capability] = default_value
            env['n_rows'] = default_value
            return env
        else:
            return {}

    def renderable_agg_bindings(self, i, default_value=None):
        return self.child.typ.entry_env(default_value) if i == 1 else {}

    def renderable_scan_bindings(self, i, default_value=None):
        return self.child.typ.col_env(default_value) if i == 1 else {}


class MatrixUnionCols(MatrixIR):
    def __init__(self, left, right, join_type):
        super().__init__(left, right)
        self.left = left
        self.right = right
        self.join_type = join_type

    def head_str(self):
        return f'{escape_id(self.join_type)}'

    def _eq(self, other):
        return self.join_type == other.join_type

    def _compute_type(self):
        self.right.typ  # force
        self._type = self.left.typ


class MatrixMapEntries(MatrixIR):
    def __init__(self, child, new_entry):
        super().__init__(child, new_entry)
        self.child = child
        self.new_entry = new_entry

    def _compute_type(self):
        child_typ = self.child.typ
        self.new_entry._compute_type(child_typ.entry_env(), None)
        self._type = hl.tmatrix(
            child_typ.global_type,
            child_typ.col_type,
            child_typ.col_key,
            child_typ.row_type,
            child_typ.row_key,
            self.new_entry.typ)

    def renderable_bindings(self, i, default_value=None):
        return self.child.typ.entry_env(default_value) if i == 1 else {}


class MatrixFilterEntries(MatrixIR):
    def __init__(self, child, pred):
        super().__init__(child, pred)
        self.child = child
        self.pred = pred

    def _compute_type(self):
        self.pred._compute_type(self.child.typ.entry_env(), None)
        self._type = self.child.typ

    def renderable_bindings(self, i, default_value=None):
        return self.child.typ.entry_env(default_value) if i == 1 else {}


class MatrixKeyRowsBy(MatrixIR):
    def __init__(self, child, keys, is_sorted=False):
        super().__init__(child)
        self.child = child
        self.keys = keys
        self.is_sorted = is_sorted

    def head_str(self):
        return '({}) {}'.format(
            ' '.join([escape_id(x) for x in self.keys]),
            self.is_sorted)

    def _eq(self, other):
        return self.keys == other.keys and self.is_sorted == other.is_sorted

    def _compute_type(self):
        child_typ = self.child.typ
        self._type = hl.tmatrix(
            child_typ.global_type,
            child_typ.col_type,
            child_typ.col_key,
            child_typ.row_type,
            self.keys,
            child_typ.entry_type)


class MatrixMapRows(MatrixIR):
    def __init__(self, child, new_row):
        super().__init__(child, new_row)
        self.child = child
        self.new_row = new_row

    def _compute_type(self):
        child_typ = self.child.typ
        self.new_row._compute_type({**child_typ.row_env(), 'n_cols': hl.tint32}, child_typ.entry_env())
        self._type = hl.tmatrix(
            child_typ.global_type,
            child_typ.col_type,
            child_typ.col_key,
            self.new_row.typ,
            child_typ.row_key,
            child_typ.entry_type)

    def renderable_bindings(self, i, default_value=None):
        if i == 1:
            env = self.child.typ.row_env(default_value)
            env[BaseIR.agg_capability] = default_value
            env['n_cols'] = default_value
            return env
        else:
            return {}

    def renderable_agg_bindings(self, i, default_value=None):
        return self.child.typ.entry_env(default_value) if i == 1 else {}

    def renderable_scan_bindings(self, i, default_value=None):
        return self.child.typ.row_env(default_value) if i == 1 else {}


class MatrixMapGlobals(MatrixIR):
    def __init__(self, child, new_global):
        super().__init__(child, new_global)
        self.child = child
        self.new_global = new_global

    def _compute_type(self):
        child_typ = self.child.typ
        self.new_global._compute_type(child_typ.global_env(), None)
        self._type = hl.tmatrix(
            self.new_global.typ,
            child_typ.col_type,
            child_typ.col_key,
            child_typ.row_type,
            child_typ.row_key,
            child_typ.entry_type)

    def renderable_bindings(self, i, default_value=None):
        return self.child.typ.global_env(default_value) if i == 1 else {}


class MatrixFilterCols(MatrixIR):
    def __init__(self, child, pred):
        super().__init__(child, pred)
        self.child = child
        self.pred = pred

    def _compute_type(self):
        self.pred._compute_type(self.child.typ.col_env(), None)
        self._type = self.child.typ

    def renderable_bindings(self, i, default_value=None):
        return self.child.typ.col_env(default_value) if i == 1 else {}


class MatrixCollectColsByKey(MatrixIR):
    def __init__(self, child):
        super().__init__(child)
        self.child = child

    def _compute_type(self):
        child_typ = self.child.typ
        self._type = hl.tmatrix(
            child_typ.global_type,
            child_typ.col_key_type._concat(
                hl.tstruct(**{f: hl.tarray(t) for f, t in child_typ.col_value_type.items()})),
            child_typ.col_key,
            child_typ.row_type,
            child_typ.row_key,
            hl.tstruct(**{f: hl.tarray(t) for f, t in child_typ.entry_type.items()}))


class MatrixAggregateColsByKey(MatrixIR):
    def __init__(self, child, entry_expr, col_expr):
        super().__init__(child, entry_expr, col_expr)
        self.child = child
        self.entry_expr = entry_expr
        self.col_expr = col_expr

    def _compute_type(self):
        child_typ = self.child.typ
        self.entry_expr._compute_type(child_typ.row_env(), child_typ.entry_env())
        self.col_expr._compute_type(child_typ.global_env(), child_typ.col_env())
        self._type = hl.tmatrix(
            child_typ.global_type,
            child_typ.col_key_type._concat(self.col_expr.typ),
            child_typ.col_key,
            child_typ.row_type,
            child_typ.row_key,
            self.entry_expr.typ)

    def renderable_bindings(self, i, default_value=None):
        if i == 1:
            env = self.child.typ.row_env(default_value)
            env[BaseIR.agg_capability] = default_value
            return env
        elif i == 2:
            env = self.child.typ.global_env(default_value)
            env[BaseIR.agg_capability] = default_value
            return env
        else:
            return {}

    def renderable_agg_bindings(self, i, default_value=None):
        if i == 1:
            return self.child.typ.entry_env(default_value)
        elif i == 2:
            return self.child.typ.col_env(default_value)
        else:
            return {}


class MatrixExplodeRows(MatrixIR):
    def __init__(self, child, path):
        super().__init__(child)
        self.child = child
        self.path = path

    def head_str(self):
        return f"({' '.join([escape_id(id) for id in self.path])})"

    def _eq(self, other):
        return self.path == other.path

    def _compute_type(self):
        child_typ = self.child.typ
        a = child_typ.row_type._index_path(self.path)
        new_row_type = child_typ.row_type._insert(self.path, a.element_type)
        self._type = hl.tmatrix(
            child_typ.global_type,
            child_typ.col_type,
            child_typ.col_key,
            new_row_type,
            child_typ.row_key,
            child_typ.entry_type)


class MatrixRepartition(MatrixIR):
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


class MatrixUnionRows(MatrixIR):
    def __init__(self, *children):
        super().__init__(*children)
        self.children = children

    def _compute_type(self):
        for c in self.children:
            c.typ  # force
        self._type = self.children[0].typ


class MatrixDistinctByRow(MatrixIR):
    def __init__(self, child):
        super().__init__(child)
        self.child = child

    def _compute_type(self):
        self._type = self.child.typ


class MatrixRowsHead(MatrixIR):
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


class MatrixColsHead(MatrixIR):
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


class MatrixRowsTail(MatrixIR):
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


class MatrixColsTail(MatrixIR):
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


class MatrixExplodeCols(MatrixIR):
    def __init__(self, child, path):
        super().__init__(child)
        self.child = child
        self.path = path

    def head_str(self):
        return f"({' '.join([escape_id(id) for id in self.path])})"

    def _eq(self, other):
        return self.path == other.path

    def _compute_type(self):
        child_typ = self.child.typ
        a = child_typ.col_type._index_path(self.path)
        new_col_type = child_typ.col_type._insert(self.path, a.element_type)
        self._type = hl.tmatrix(
            child_typ.global_type,
            new_col_type,
            child_typ.col_key,
            child_typ.row_type,
            child_typ.row_key,
            child_typ.entry_type)


class CastTableToMatrix(MatrixIR):
    def __init__(self, child, entries_field_name, cols_field_name, col_key):
        super().__init__(child)
        self.child = child
        self.entries_field_name = entries_field_name
        self.cols_field_name = cols_field_name
        self.col_key = col_key

    def head_str(self):
        return '{} {} ({})'.format(
            escape_str(self.entries_field_name),
            escape_str(self.cols_field_name),
            ' '.join([escape_id(id) for id in self.col_key]))

    def _eq(self, other):
        return self.entries_field_name == other.entries_field_name and \
            self.cols_field_name == other.cols_field_name and \
            self.col_key == other.col_key

    def _compute_type(self):
        child_typ = self.child.typ
        self._type = hl.tmatrix(
            child_typ.global_type._drop_fields([self.cols_field_name]),
            child_typ.global_type[self.cols_field_name].element_type,
            self.col_key,
            child_typ.row_type._drop_fields([self.entries_field_name]),
            child_typ.row_key,
            child_typ.row_type[self.entries_field_name].element_type)


class MatrixAnnotateRowsTable(MatrixIR):
    def __init__(self, child, table, root, product=False):
        super().__init__(child, table)
        self.child = child
        self.table = table
        self.root = root
        self.product = product

    def head_str(self):
        return f'"{escape_str(self.root)}" {self.product}'

    def _eq(self, other):
        return self.root == other.root and self.product == other.product

    def _compute_type(self):
        child_typ = self.child.typ
        if self.product:
            value_type = hl.tarray(self.table.typ.value_type)
        else:
            value_type = self.table.typ.value_type
        self._type = hl.tmatrix(
            child_typ.global_type,
            child_typ.col_type,
            child_typ.col_key,
            child_typ.row_type._insert_field(self.root, value_type),
            child_typ.row_key,
            child_typ.entry_type)


class MatrixAnnotateColsTable(MatrixIR):
    def __init__(self, child, table, root):
        super().__init__(child, table)
        self.child = child
        self.table = table
        self.root = root

    def head_str(self):
        return f'"{escape_str(self.root)}"'

    def _eq(self, other):
        return self.root == other.root

    def _compute_type(self):
        child_typ = self.child.typ
        self._type = hl.tmatrix(
            child_typ.global_type,
            child_typ.col_type._insert_field(self.root, self.table.typ.value_type),
            child_typ.col_key,
            child_typ.row_type,
            child_typ.row_key,
            child_typ.entry_type)


class MatrixToMatrixApply(MatrixIR):
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
        if name == 'MatrixFilterPartitions':
            self._type = child_typ


class MatrixRename(MatrixIR):
    def __init__(self, child, global_map, col_map, row_map, entry_map):
        super().__init__(child)
        self.child = child
        self.global_map = global_map
        self.col_map = col_map
        self.row_map = row_map
        self.entry_map = entry_map

    def head_str(self):
        return f'{parsable_strings(self.global_map.keys())} ' \
               f'{parsable_strings(self.global_map.values())} ' \
               f'{parsable_strings(self.col_map.keys())} ' \
               f'{parsable_strings(self.col_map.values())} ' \
               f'{parsable_strings(self.row_map.keys())} ' \
               f'{parsable_strings(self.row_map.values())} ' \
               f'{parsable_strings(self.entry_map.keys())} ' \
               f'{parsable_strings(self.entry_map.values())} '

    def _eq(self, other):
        return self.global_map == other.global_map and \
            self.col_map == other.col_map and \
            self.row_map == other.row_map and \
            self.entry_map == other.entry_map

    def _compute_type(self):
        self._type = self.child.typ._rename(self.global_map, self.col_map, self.row_map, self.entry_map)


class MatrixFilterIntervals(MatrixIR):
    def __init__(self, child, intervals, point_type, keep):
        super().__init__(child)
        self.child = child
        self.intervals = intervals
        self.point_type = point_type
        self.keep = keep

    def head_str(self):
        return f'{dump_json(hl.tarray(hl.tinterval(self.point_type))._convert_to_json(self.intervals))} {self.keep}'

    def _eq(self, other):
        return self.intervals == other.intervals and self.point_type == other.point_type and self.keep == other.keep

    def _compute_type(self):
        self._type = self.child.typ


class JavaMatrix(MatrixIR):
    def __init__(self, jir):
        super().__init__()
        self._jir = jir

    def render_head(self, r):
        return f'(JavaMatrix {r.add_jir(self._jir)}'

    def _compute_type(self):
        self._type = hl.tmatrix._from_java(self._jir.typ())


class JavaMatrixVectorRef(MatrixIR):
    def __init__(self, vec_ref, idx):
        super().__init__()
        self.vec_ref = vec_ref
        self.idx = idx

    def head_str(self):
        return f'{self.vec_ref.jid} {self.idx}'

    def _compute_type(self):
        self._type = self.vec_ref.item_type
