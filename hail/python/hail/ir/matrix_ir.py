from typing import Optional
import hail as hl
from hail.expr.types import HailType, tint64
from hail.ir.base_ir import BaseIR, MatrixIR
from hail.ir.utils import modify_deep_field, zip_with_index, zip_with_index_field, default_row_uid, default_col_uid, unpack_row_uid, unpack_col_uid
import hail.ir.ir as ir
from hail.utils import FatalError
from hail.utils.misc import escape_str, parsable_strings, escape_id
from hail.utils.jsonx import dump_json
from hail.utils.java import Env


class MatrixAggregateRowsByKey(MatrixIR):
    def __init__(self, child, entry_expr, row_expr):
        super().__init__(child, entry_expr, row_expr)
        self.child = child
        self.entry_expr = entry_expr
        self.row_expr = row_expr

    def _handle_randomness(self, row_uid_field_name, col_uid_field_name):
        drop_row_uid = False
        drop_col_uid = False
        if self.entry_expr.uses_randomness or self.row_expr.uses_randomness:
            drop_row_uid = row_uid_field_name is None
            if row_uid_field_name is None:
                row_uid_field_name = default_row_uid
        if self.entry_expr.uses_randomness:
            drop_col_uid = col_uid_field_name is None
            if col_uid_field_name is None:
                col_uid_field_name = default_col_uid

        child = self.child.handle_randomness(row_uid_field_name, col_uid_field_name)
        entry_expr = self.entry_expr
        row_expr = self.row_expr
        if row_uid_field_name is not None:
            row_uid, old_row = unpack_row_uid(child.typ.row_type, row_uid_field_name)
            first_row_uid = ir.ArrayRef(ir.ApplyAggOp('Take', [ir.I32(1)], [row_uid]), ir.I32(0))
            entry_expr = ir.AggLet('va', old_row, entry_expr, is_scan=False)
            row_expr = ir.AggLet('va', old_row, row_expr, is_scan=False)
            row_expr = ir.InsertFields(row_expr, [(row_uid_field_name, first_row_uid)], None)
        if col_uid_field_name is not None:
            col_uid, old_col = unpack_col_uid(child.typ.col_type, col_uid_field_name)
            entry_expr = ir.AggLet('sa', old_col, entry_expr, is_scan=False)
            entry_expr = ir.Let('sa', old_col, entry_expr)
        if self.entry_expr.uses_value_randomness:
            entry_expr = ir.Let('__rng_state',
                                ir.RNGSplit(ir.RNGStateLiteral(), ir.concat_uids(first_row_uid, col_uid)),
                                entry_expr)
        if self.entry_expr.uses_agg_randomness(is_scan=False):
            entry_expr = ir.AggLet('__rng_state',
                                   ir.RNGSplit(ir.RNGStateLiteral(), ir.concat_uids(row_uid, col_uid)),
                                   entry_expr,
                                   is_scan=False)
        if self.row_expr.uses_value_randomness:
            row_expr = ir.Let('__rng_state',
                              ir.RNGSplit(ir.RNGStateLiteral(), first_row_uid),
                              row_expr)
        if self.row_expr.uses_agg_randomness(is_scan=False):
            row_expr = ir.AggLet('__rng_state',
                                 ir.RNGSplit(ir.RNGStateLiteral(), row_uid),
                                 row_expr,
                                 is_scan=False)

        result = MatrixAggregateRowsByKey(child, entry_expr, row_expr)
        if drop_row_uid:
            _, old_row = unpack_row_uid(result.typ.row_type, row_uid_field_name)
            result = MatrixMapRows(result, old_row)
        if drop_col_uid:
            _, old_col = unpack_col_uid(result.typ.col_type, col_uid_field_name)
            result = MatrixMapCols(result, old_col, None)
        return result

    def _compute_type(self, deep_typecheck):
        self.child.compute_type(deep_typecheck)
        child_typ = self.child.typ
        self.entry_expr.compute_type(child_typ.col_env(), child_typ.entry_env(), deep_typecheck)
        self.row_expr.compute_type(child_typ.global_env(), child_typ.row_env(), deep_typecheck)
        return hl.tmatrix(
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
    def __init__(self,
                 reader,
                 drop_cols: bool = False,
                 drop_rows: bool = False,
                 drop_row_uids: bool = True,
                 drop_col_uids: bool = True,
                 *,
                 _assert_type: Optional[HailType] = None):
        super().__init__()
        self.reader = reader
        self.drop_cols = drop_cols
        self.drop_rows = drop_rows
        self.drop_row_uids = drop_row_uids
        self.drop_col_uids = drop_col_uids
        self._type: Optional[HailType] = _assert_type

    def _handle_randomness(self, row_uid_field_name, col_uid_field_name):
        rename_row_uid = False
        rename_col_uid = False
        drop_row_uids = False
        drop_col_uids = False
        if row_uid_field_name is None and self.drop_row_uids:
            drop_row_uids = True
        elif row_uid_field_name is not None and row_uid_field_name != default_row_uid:
            rename_row_uid = True
        if col_uid_field_name is None and self.drop_col_uids:
            drop_col_uids = True
        elif col_uid_field_name is not None and col_uid_field_name != default_col_uid:
            rename_col_uid = True
        result = MatrixRead(self.reader, self.drop_cols, self.drop_rows, drop_row_uids, drop_col_uids)
        if rename_row_uid or rename_col_uid:
            rename = False
            row_map = {}
            col_map = {}
            if rename_row_uid:
                if self.drop_row_uids:
                    rename = True
                    row_map = {default_row_uid: row_uid_field_name}
                else:
                    row = ir.Ref('va', self.typ.row_type)
                    result = MatrixMapRows(
                        result,
                        ir.InsertFields(row, [(row_uid_field_name, ir.GetField(row, default_row_uid))], None))
            if rename_col_uid:
                if self.drop_col_uids:
                    rename = True
                    col_map = {default_col_uid: col_uid_field_name}
                else:
                    col = ir.Ref('sa', self.typ.col_type)
                    result = MatrixMapCols(
                        result,
                        ir.InsertFields(col, [(col_uid_field_name, ir.GetField(col, default_col_uid))], None),
                        None)
            if rename:
                result = MatrixRename(result, {}, col_map, row_map, {})
        return result

    def render_head(self, r):
        if self.drop_row_uids and self.drop_col_uids:
            reqType = "DropRowColUIDs"
        elif self.drop_row_uids:
            reqType = "DropRowUIDs"
        elif self.drop_col_uids:
            reqType = "DropColUIDs"
        else:
            reqType = "None"
        return f'(MatrixRead {reqType} {self.drop_cols} {self.drop_rows} "{self.reader.render(r)}"'

    def _eq(self, other):
        return (self.reader == other.reader
                and self.drop_cols == other.drop_cols
                and self.drop_rows == other.drop_rows
                and self.drop_row_uids == other.drop_row_uids
                and self.drop_col_uids == other.drop_col_uids)

    def _compute_type(self, deep_typecheck):
        if self._type is None:
            return Env.backend().matrix_type(self)
        else:
            return self._type


class MatrixFilterRows(MatrixIR):
    def __init__(self, child, pred):
        super().__init__(child, pred)
        self.child = child
        self.pred = pred

    def _handle_randomness(self, row_uid_field_name, col_uid_field_name):
        if not self.pred.uses_randomness and row_uid_field_name is None:
            child = self.child.handle_randomness(None, col_uid_field_name)
            return MatrixFilterRows(child, self.pred)

        drop_row_uid = row_uid_field_name is None and default_row_uid not in self.child.typ.row_type
        if row_uid_field_name is None:
            row_uid_field_name = default_row_uid
        child = self.child.handle_randomness(row_uid_field_name, col_uid_field_name)
        uid, old_row = unpack_row_uid(child.typ.row_type, row_uid_field_name)
        pred = ir.Let('va', old_row, self.pred)
        if self.pred.uses_randomness:
            pred = ir.Let('__rng_state', ir.RNGSplit(ir.RNGStateLiteral(), uid), pred)
        result = MatrixFilterRows(child, pred)
        if drop_row_uid:
            result = MatrixMapRows(result, old_row)
        return result

    def _compute_type(self, deep_typecheck):
        self.child.compute_type(deep_typecheck)
        self.pred.compute_type(self.child.typ.row_env(), None, deep_typecheck)
        return self.child.typ

    def renderable_bindings(self, i, default_value=None):
        return self.child.typ.row_env(default_value) if i == 1 else {}


class MatrixChooseCols(MatrixIR):
    def __init__(self, child, old_indices):
        super().__init__(child)
        self.child = child
        self.old_indices = old_indices

    def _handle_randomness(self, row_uid_field_name, col_uid_field_name):
        child = self.child.handle_randomness(row_uid_field_name, col_uid_field_name)
        return MatrixChooseCols(child, self.old_indices)

    def head_str(self):
        return f'({" ".join([str(i) for i in self.old_indices])})'

    def _eq(self, other):
        return self.old_indices == other.old_indices

    def _compute_type(self, deep_typecheck):
        self.child.compute_type(deep_typecheck)
        return self.child.typ


class MatrixMapCols(MatrixIR):
    def __init__(self, child, new_col, new_key):
        super().__init__(child, new_col)
        self.child = child
        self.new_col = new_col
        self.new_key = new_key

    def _handle_randomness(self, row_uid_field_name, col_uid_field_name):
        if not self.new_col.uses_randomness and col_uid_field_name is None:
            child = self.child.handle_randomness(row_uid_field_name, None)
            return MatrixMapCols(child, self.new_col, self.new_key)

        drop_row_uid = False
        if self.new_col.uses_agg_randomness(is_scan=False) and row_uid_field_name is None:
            drop_row_uid = True
            row_uid_field_name = default_row_uid
        keep_col_uid = col_uid_field_name is not None
        if col_uid_field_name is None:
            col_uid_field_name = default_col_uid
        child = self.child.handle_randomness(row_uid_field_name, col_uid_field_name)
        col_uid, old_col = unpack_col_uid(child.typ.col_type, col_uid_field_name)
        new_col = ir.Let('sa', old_col, self.new_col)
        if row_uid_field_name is not None:
            row_uid, old_row = unpack_row_uid(child.typ.row_type, row_uid_field_name)
        if self.new_col.uses_value_randomness:
            new_col = ir.Let('__rng_state', ir.RNGSplit(ir.RNGStateLiteral(), col_uid), new_col)
        if self.new_col.uses_agg_randomness(is_scan=True):
            new_col = ir.AggLet('__rng_state', ir.RNGSplit(ir.RNGStateLiteral(), col_uid), new_col, is_scan=True)
        if self.new_col.uses_agg_randomness(is_scan=False):
            entry_uid = ir.concat_uids(row_uid, col_uid)
            new_col = ir.AggLet('__rng_state', ir.RNGSplit(ir.RNGStateLiteral(), entry_uid), new_col, is_scan=False)
        if keep_col_uid:
            new_col = ir.InsertFields(new_col, [(col_uid_field_name, col_uid)], None)
        result = MatrixMapCols(child, new_col, self.new_key)
        if drop_row_uid:
            result = MatrixMapRows(result, old_row)
        return result

    def head_str(self):
        return '(' + ' '.join(f'"{escape_str(f)}"' for f in self.new_key) + ')' if self.new_key is not None else 'None'

    def _eq(self, other):
        return self.new_key == other.new_key

    def _compute_type(self, deep_typecheck):
        self.child.compute_type(deep_typecheck)
        child_typ = self.child.typ
        self.new_col.compute_type({**child_typ.col_env(), 'n_rows': hl.tint64}, child_typ.entry_env(), deep_typecheck)
        return hl.tmatrix(
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

    def _handle_randomness(self, row_uid_field_name, col_uid_field_name):
        if self.join_type == 'outer' and row_uid_field_name is not None:
            right_row_uid_field_name = f'{row_uid_field_name}_right'
        else:
            right_row_uid_field_name = None
        left = self.left.handle_randomness(row_uid_field_name, col_uid_field_name)
        right = self.right.handle_randomness(right_row_uid_field_name, col_uid_field_name)

        if col_uid_field_name is not None:
            (left_uid, _) = unpack_col_uid(left.typ.col_type, col_uid_field_name)
            (right_uid, _) = unpack_col_uid(right.typ.col_type, col_uid_field_name)
            uid_type = ir.unify_uid_types((left_uid.typ, right_uid.typ), tag=True)
            left = MatrixMapCols(left,
                                 ir.InsertFields(ir.Ref('sa', left.typ.col_type),
                                                 [(col_uid_field_name, ir.pad_uid(left_uid, uid_type, 0))], None),
                                 new_key=None)
            right = MatrixMapCols(right,
                                  ir.InsertFields(ir.Ref('sa', right.typ.col_type),
                                                  [(col_uid_field_name, ir.pad_uid(right_uid, uid_type, 1))], None),
                                  new_key=None)

        result = MatrixUnionCols(left, right, self.join_type)

        if row_uid_field_name is not None and right_row_uid_field_name is not None:
            row = ir.Ref('row', result.typ.row_type)
            old_joined_row = ir.SelectFields(row, [field for field in self.typ.row_type])
            left_uid = ir.GetField(row, row_uid_field_name)
            right_uid = ir.GetField(row, right_row_uid_field_name)
            uid = ir.concat_uids(left_uid, right_uid, True, True)
            new_row = ir.InsertFields(old_joined_row, [(row_uid_field_name, uid)], None)
            result = MatrixMapRows(result, new_row)

        return result

    def head_str(self):
        return f'{escape_id(self.join_type)}'

    def _eq(self, other):
        return self.join_type == other.join_type

    def _compute_type(self, deep_typecheck):
        self.left.compute_type(deep_typecheck)
        self.right.compute_type(deep_typecheck)
        left_typ = self.left.typ
        right_typ = self.right.typ
        return hl.tmatrix(
            global_type=left_typ.global_type,
            col_type=left_typ.col_type,
            col_key=left_typ.col_key,
            row_type=left_typ.row_type._concat(right_typ.row_value_type),
            row_key=left_typ.row_key,
            entry_type=left_typ.entry_type)


class MatrixMapEntries(MatrixIR):
    def __init__(self, child, new_entry):
        super().__init__(child, new_entry)
        self.child = child
        self.new_entry = new_entry

    def _handle_randomness(self, row_uid_field_name, col_uid_field_name):
        drop_row_uid = False
        drop_col_uid = False
        if self.new_entry.uses_randomness:
            drop_row_uid = row_uid_field_name is None
            drop_col_uid = col_uid_field_name is None
            if row_uid_field_name is None:
                row_uid_field_name = default_row_uid
            if col_uid_field_name is None:
                col_uid_field_name = default_col_uid

        child = self.child.handle_randomness(row_uid_field_name, col_uid_field_name)
        new_entry = self.new_entry
        if row_uid_field_name is not None:
            row_uid, old_row = unpack_row_uid(child.typ.row_type, row_uid_field_name)
            new_entry = ir.Let('va', old_row, new_entry)
        if col_uid_field_name is not None:
            col_uid, old_col = unpack_col_uid(child.typ.col_type, col_uid_field_name)
            new_entry = ir.Let('sa', old_col, new_entry)
        if self.new_entry.uses_value_randomness:
            new_entry = ir.Let('__rng_state', ir.RNGSplit(ir.RNGStateLiteral(), ir.concat_uids(row_uid, col_uid)), new_entry)
        result = MatrixMapEntries(child, new_entry)
        if drop_row_uid:
            _, old_row = unpack_row_uid(result.typ.row_type, row_uid_field_name)
            result = MatrixMapRows(result, old_row)
        if drop_col_uid:
            _, old_col = unpack_col_uid(result.typ.col_type, col_uid_field_name)
            result = MatrixMapCols(result, old_col, None)
        return result

    def _compute_type(self, deep_typecheck):
        self.child.compute_type(deep_typecheck)
        child_typ = self.child.typ
        self.new_entry.compute_type(child_typ.entry_env(), None, deep_typecheck)
        return hl.tmatrix(
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

    def _handle_randomness(self, row_uid_field_name, col_uid_field_name):
        drop_row_uid = False
        drop_col_uid = False
        if self.pred.uses_randomness:
            drop_row_uid = row_uid_field_name is None
            drop_col_uid = col_uid_field_name is None
            if row_uid_field_name is None:
                row_uid_field_name = default_row_uid
            if col_uid_field_name is None:
                col_uid_field_name = default_col_uid

        child = self.child.handle_randomness(row_uid_field_name, col_uid_field_name)
        pred = self.pred
        if row_uid_field_name is not None:
            row_uid, old_row = unpack_row_uid(child.typ.row_type, row_uid_field_name)
            pred = ir.Let('va', old_row, pred)
        if col_uid_field_name is not None:
            col_uid, old_col = unpack_col_uid(child.typ.col_type, col_uid_field_name)
            pred = ir.Let('sa', old_col, pred)
        if self.pred.uses_value_randomness:
            pred = ir.Let('__rng_state', ir.RNGSplit(ir.RNGStateLiteral(), ir.concat_uids(row_uid, col_uid)), pred)
        result = MatrixFilterEntries(child, pred)
        if drop_row_uid:
            _, old_row = unpack_row_uid(result.typ.row_type, row_uid_field_name)
            result = MatrixMapRows(result, old_row)
        if drop_col_uid:
            _, old_col = unpack_col_uid(result.typ.col_type, col_uid_field_name)
            result = MatrixMapCols(result, old_col, None)
        return result

    def _compute_type(self, deep_typecheck):
        self.child.compute_type(deep_typecheck)
        self.pred.compute_type(self.child.typ.entry_env(), None, deep_typecheck)
        return self.child.typ

    def renderable_bindings(self, i, default_value=None):
        return self.child.typ.entry_env(default_value) if i == 1 else {}


class MatrixKeyRowsBy(MatrixIR):
    def __init__(self, child, keys, is_sorted=False):
        super().__init__(child)
        self.child = child
        self.keys = keys
        self.is_sorted = is_sorted

    def _handle_randomness(self, row_uid_field_name, col_uid_field_name):
        child = self.child.handle_randomness(row_uid_field_name, col_uid_field_name)
        return MatrixKeyRowsBy(child, self.keys, self.is_sorted)

    def head_str(self):
        return '({}) {}'.format(
            ' '.join([escape_id(x) for x in self.keys]),
            self.is_sorted)

    def _eq(self, other):
        return self.keys == other.keys and self.is_sorted == other.is_sorted

    def _compute_type(self, deep_typecheck):
        self.child.compute_type(deep_typecheck)
        child_typ = self.child.typ
        return hl.tmatrix(
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

    def _handle_randomness(self, row_uid_field_name, col_uid_field_name):
        if not self.new_row.uses_randomness and row_uid_field_name is None:
            child = self.child.handle_randomness(None, col_uid_field_name)
            return MatrixMapRows(child, self.new_row)

        drop_col_uid = False
        if self.new_row.uses_agg_randomness(is_scan=False) and col_uid_field_name is None:
            col_uid_field_name = default_col_uid
            drop_col_uid = True
        keep_row_uid = row_uid_field_name is not None
        if row_uid_field_name is None:
            row_uid_field_name = default_row_uid
        child = self.child.handle_randomness(row_uid_field_name, col_uid_field_name)
        row_uid, old_row = unpack_row_uid(child.typ.row_type, row_uid_field_name,
                                          drop_uid=row_uid_field_name not in self.child.typ.row_type)
        new_row = ir.Let('va', old_row, self.new_row)
        if col_uid_field_name is not None:
            col_uid, old_col = unpack_col_uid(child.typ.col_type, col_uid_field_name)
        if self.new_row.uses_value_randomness:
            new_row = ir.Let('__rng_state', ir.RNGSplit(ir.RNGStateLiteral(), row_uid), new_row)
        if self.new_row.uses_agg_randomness(is_scan=True):
            new_row = ir.AggLet('__rng_state', ir.RNGSplit(ir.RNGStateLiteral(), row_uid), new_row, is_scan=True)
        if self.new_row.uses_agg_randomness(is_scan=False):
            entry_uid = ir.concat_uids(row_uid, col_uid)
            new_row = ir.AggLet('__rng_state', ir.RNGSplit(ir.RNGStateLiteral(), entry_uid), new_row, is_scan=False)
        if keep_row_uid:
            new_row = ir.InsertFields(new_row, [(row_uid_field_name, row_uid)], None)
        result = MatrixMapRows(child, new_row)
        if drop_col_uid:
            result = MatrixMapCols(result, old_col, None)
        return result

    def _compute_type(self, deep_typecheck):
        self.child.compute_type(deep_typecheck)
        child_typ = self.child.typ
        self.new_row.compute_type({**child_typ.row_env(), 'n_cols': hl.tint32}, child_typ.entry_env(), deep_typecheck)
        return hl.tmatrix(
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

    def _handle_randomness(self, row_uid_field_name, col_uid_field_name):
        child = self.child.handle_randomness(row_uid_field_name, col_uid_field_name)
        new_global = self.new_global
        if new_global.uses_randomness:
            new_global = ir.Let('__rng_state', ir.RNGStateLiteral(), new_global)
        return MatrixMapGlobals(child, new_global)

    def _compute_type(self, deep_typecheck):
        self.child.compute_type(deep_typecheck)
        child_typ = self.child.typ
        self.new_global.compute_type(child_typ.global_env(), None, deep_typecheck)
        return hl.tmatrix(
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

    def _handle_randomness(self, row_uid_field_name, col_uid_field_name):
        if not self.pred.uses_randomness and col_uid_field_name is None:
            child = self.child.handle_randomness(row_uid_field_name, None)
            return MatrixFilterCols(child, self.pred)

        drop_col_uid = col_uid_field_name is None
        if col_uid_field_name is None:
            col_uid_field_name = default_col_uid
        child = self.child.handle_randomness(row_uid_field_name, col_uid_field_name)
        col_uid, old_col = unpack_col_uid(child.typ.col_type, col_uid_field_name)
        pred = ir.Let('sa', old_col, self.pred)
        if self.pred.uses_randomness:
            pred = ir.Let('__rng_state', ir.RNGSplit(ir.RNGStateLiteral(), col_uid), pred)
        result = MatrixFilterCols(child, pred)
        if drop_col_uid:
            result = MatrixMapCols(result, old_col, new_key=None)
        return result

    def _compute_type(self, deep_typecheck):
        self.child.compute_type(deep_typecheck)
        self.pred.compute_type(self.child.typ.col_env(), None, deep_typecheck)
        return self.child.typ

    def renderable_bindings(self, i, default_value=None):
        return self.child.typ.col_env(default_value) if i == 1 else {}


class MatrixCollectColsByKey(MatrixIR):
    def __init__(self, child):
        super().__init__(child)
        self.child = child

    def _handle_randomness(self, row_uid_field_name, col_uid_field_name):
        child = self.child.handle_randomness(row_uid_field_name, col_uid_field_name)
        result = MatrixCollectColsByKey(child)
        if col_uid_field_name is not None:
            col = ir.Ref('sa', result.typ.col_type)
            uids = ir.GetField(col, col_uid_field_name)
            # FIXME: might cause issues being dependent on col order
            uid = ir.ArrayRef(uids, ir.I32(0))
            result = MatrixMapCols(result, ir.InsertFields(col, [(col_uid_field_name, uid)], None), None)
        return result

    def _compute_type(self, deep_typecheck):
        self.child.compute_type(deep_typecheck)
        child_typ = self.child.typ
        return hl.tmatrix(
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

    def _handle_randomness(self, row_uid_field_name, col_uid_field_name):
        drop_row_uid = False
        drop_col_uid = False
        if self.entry_expr.uses_randomness:
            drop_row_uid = row_uid_field_name is None
            if row_uid_field_name is None:
                row_uid_field_name = default_row_uid
        if self.entry_expr.uses_randomness or self.col_expr.uses_randomness:
            drop_col_uid = col_uid_field_name is None
            if col_uid_field_name is None:
                col_uid_field_name = default_col_uid

        child = self.child.handle_randomness(row_uid_field_name, col_uid_field_name)
        entry_expr = self.entry_expr
        col_expr = self.col_expr
        if row_uid_field_name is not None:
            row_uid, old_row = unpack_row_uid(child.typ.row_type, row_uid_field_name)
            entry_expr = ir.Let('va', old_row, entry_expr)
            entry_expr = ir.AggLet('va', old_row, entry_expr, is_scan=False)
        if col_uid_field_name is not None:
            col_uid, old_col = unpack_col_uid(child.typ.col_type, col_uid_field_name)
            first_col_uid = ir.ArrayRef(ir.ApplyAggOp('Take', [ir.I32(1)], [col_uid]), ir.I32(0))
            entry_expr = ir.AggLet('sa', old_col, entry_expr, is_scan=False)
            col_expr = ir.AggLet('sa', old_col, col_expr, is_scan=False)
            col_expr = ir.InsertFields(col_expr, [(col_uid_field_name, first_col_uid)], None)
        if self.entry_expr.uses_value_randomness:
            entry_expr = ir.Let('__rng_state',
                                ir.RNGSplit(ir.RNGStateLiteral(),
                                            ir.concat_uids(row_uid, first_col_uid)),
                                entry_expr)
        if self.entry_expr.uses_agg_randomness(is_scan=False):
            entry_expr = ir.AggLet('__rng_state',
                                   ir.RNGSplit(ir.RNGStateLiteral(),
                                               ir.concat_uids(row_uid, col_uid)),
                                   entry_expr,
                                   is_scan=False)
        if self.col_expr.uses_value_randomness:
            col_expr = ir.Let('__rng_state',
                              ir.RNGSplit(ir.RNGStateLiteral(), first_col_uid),
                              col_expr)
        if self.col_expr.uses_agg_randomness(is_scan=False):
            col_expr = ir.AggLet('__rng_state',
                                 ir.RNGSplit(ir.RNGStateLiteral(), col_uid),
                                 col_expr,
                                 is_scan=False)

        result = MatrixAggregateColsByKey(child, entry_expr, col_expr)
        if drop_row_uid:
            _, old_row = unpack_row_uid(result.typ.row_type, row_uid_field_name)
            result = MatrixMapRows(result, old_row)
        if drop_col_uid:
            _, old_col = unpack_col_uid(result.typ.col_type, col_uid_field_name)
            result = MatrixMapCols(result, old_col, None)
        return result

    def _compute_type(self, deep_typecheck):
        self.child.compute_type(deep_typecheck)
        child_typ = self.child.typ
        self.entry_expr.compute_type(child_typ.row_env(), child_typ.entry_env(), deep_typecheck)
        self.col_expr.compute_type(child_typ.global_env(), child_typ.col_env(), deep_typecheck)
        return hl.tmatrix(
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

    def _handle_randomness(self, row_uid_field_name, col_uid_field_name):
        if row_uid_field_name is None:
            MatrixExplodeRows(self.child.handle_randomness(None, col_uid_field_name), self.path)

        child = self.child.handle_randomness(row_uid_field_name, col_uid_field_name)

        if row_uid_field_name not in child.typ.row_type.fields:
            return MatrixExplodeRows(child, self.path)

        new_row = modify_deep_field(ir.Ref('va', child.typ.row_type), self.path, zip_with_index)
        child = MatrixMapRows(child, new_row)

        new_explode = MatrixExplodeRows(child, self.path)
        new_row = modify_deep_field(
            ir.Ref('va', new_explode.typ.row_type),
            self.path,
            lambda tuple: ir.GetTupleElement(tuple, 0),
            lambda row, tuple: ir.InsertFields(row, [(row_uid_field_name, ir.concat_uids(ir.GetField(row, row_uid_field_name), ir.Cast(ir.GetTupleElement(tuple, 1), tint64)))], None))
        return MatrixMapRows(new_explode, new_row)

    def head_str(self):
        return f"({' '.join([escape_id(id) for id in self.path])})"

    def _eq(self, other):
        return self.path == other.path

    def _compute_type(self, deep_typecheck):
        self.child.compute_type(deep_typecheck)
        child_typ = self.child.typ
        a = child_typ.row_type._index_path(self.path)
        new_row_type = child_typ.row_type._insert(self.path, a.element_type)
        return hl.tmatrix(
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

    def _handle_randomness(self, row_uid_field_name, col_uid_field_name):
        return MatrixRepartition(self.child.handle_randomness(row_uid_field_name, col_uid_field_name), self.n, self.strategy)

    def head_str(self):
        return f'{self.n} {self.strategy}'

    def _eq(self, other):
        return self.n == other.n and self.strategy == other.strategy

    def _compute_type(self, deep_typecheck):
        self.child.compute_type(deep_typecheck)
        return self.child.typ


class MatrixUnionRows(MatrixIR):
    def __init__(self, *children):
        super().__init__(*children)
        self.children = children

    def _handle_randomness(self, row_uid_field_name, col_uid_field_name):
        children = [self.children[0].handle_randomness(row_uid_field_name, col_uid_field_name),
                    *[child.handle_randomness(row_uid_field_name, None) for child in self.children[1:]]]

        if row_uid_field_name is not None:
            uids = [uid for uid, _ in (unpack_row_uid(child.typ.row_type, row_uid_field_name) for child in children)]
            uid_type = ir.unify_uid_types((uid.typ for uid in uids), tag=True)
            children = [MatrixMapRows(child,
                                      ir.InsertFields(ir.Ref('va', child.typ.row_type),
                                                      [(row_uid_field_name, ir.pad_uid(uid, uid_type, i))],
                                                      None))
                        for i, (child, uid) in enumerate(zip(children, uids))]

        return MatrixUnionRows(*children)

    def _compute_type(self, deep_typecheck):
        for c in self.children:
            c.compute_type(deep_typecheck)
        return self.children[0].typ


class MatrixDistinctByRow(MatrixIR):
    def __init__(self, child):
        super().__init__(child)
        self.child = child

    def _handle_randomness(self, row_uid_field_name, col_uid_field_name):
        return MatrixDistinctByRow(self.child.handle_randomness(row_uid_field_name, col_uid_field_name))

    def _compute_type(self, deep_typecheck):
        self.child.compute_type(deep_typecheck)
        return self.child.typ


class MatrixRowsHead(MatrixIR):
    def __init__(self, child, n):
        super().__init__(child)
        self.child = child
        self.n = n

    def _handle_randomness(self, row_uid_field_name, col_uid_field_name):
        return MatrixRowsHead(self.child.handle_randomness(row_uid_field_name, col_uid_field_name), self.n)

    def head_str(self):
        return self.n

    def _eq(self, other):
        return self.n == other.n

    def _compute_type(self, deep_typecheck):
        self.child.compute_type(deep_typecheck)
        return self.child.typ


class MatrixColsHead(MatrixIR):
    def __init__(self, child, n):
        super().__init__(child)
        self.child = child
        self.n = n

    def _handle_randomness(self, row_uid_field_name, col_uid_field_name):
        return MatrixColsHead(self.child.handle_randomness(row_uid_field_name, col_uid_field_name), self.n)

    def head_str(self):
        return self.n

    def _eq(self, other):
        return self.n == other.n

    def _compute_type(self, deep_typecheck):
        self.child.compute_type(deep_typecheck)
        return self.child.typ


class MatrixRowsTail(MatrixIR):
    def __init__(self, child, n):
        super().__init__(child)
        self.child = child
        self.n = n

    def _handle_randomness(self, row_uid_field_name, col_uid_field_name):
        return MatrixRowsTail(self.child.handle_randomness(row_uid_field_name, col_uid_field_name), self.n)

    def head_str(self):
        return self.n

    def _eq(self, other):
        return self.n == other.n

    def _compute_type(self, deep_typecheck):
        self.child.compute_type(deep_typecheck)
        return self.child.typ


class MatrixColsTail(MatrixIR):
    def __init__(self, child, n):
        super().__init__(child)
        self.child = child
        self.n = n

    def _handle_randomness(self, row_uid_field_name, col_uid_field_name):
        return MatrixColsTail(self.child.handle_randomness(row_uid_field_name, col_uid_field_name), self.n)

    def head_str(self):
        return self.n

    def _eq(self, other):
        return self.n == other.n

    def _compute_type(self, deep_typecheck):
        self.child.compute_type(deep_typecheck)
        return self.child.typ


class MatrixExplodeCols(MatrixIR):
    def __init__(self, child, path):
        super().__init__(child)
        self.child = child
        self.path = path

    def _handle_randomness(self, row_uid_field_name, col_uid_field_name):
        if col_uid_field_name is None:
            MatrixExplodeCols(self.child.handle_randomness(row_uid_field_name, None), self.path)

        child = self.child.handle_randomness(row_uid_field_name, col_uid_field_name)

        if col_uid_field_name not in child.typ.col_type.fields:
            return MatrixExplodeCols(child, self.path)

        new_col = modify_deep_field(ir.Ref('sa', child.typ.col_type), self.path, zip_with_index)
        child = MatrixMapCols(child, new_col, None)

        new_explode = MatrixExplodeCols(child, self.path)
        new_col = modify_deep_field(
            ir.Ref('sa', new_explode.typ.col_type),
            self.path,
            lambda tuple: ir.GetTupleElement(tuple, 0),
            lambda col, tuple: ir.InsertFields(col, [(col_uid_field_name, ir.concat_uids(ir.GetField(col, col_uid_field_name), ir.Cast(ir.GetTupleElement(tuple, 1), tint64)))], None))
        return MatrixMapCols(new_explode, new_col, None)

    def head_str(self):
        return f"({' '.join([escape_id(id) for id in self.path])})"

    def _eq(self, other):
        return self.path == other.path

    def _compute_type(self, deep_typecheck):
        self.child.compute_type(deep_typecheck)
        child_typ = self.child.typ
        a = child_typ.col_type._index_path(self.path)
        new_col_type = child_typ.col_type._insert(self.path, a.element_type)
        return hl.tmatrix(
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

    def _handle_randomness(self, row_uid_field_name, col_uid_field_name):
        from hail.ir.table_ir import TableMapGlobals
        child = self.child
        if col_uid_field_name is not None:
            new_globals = modify_deep_field(
                ir.Ref('global', child.typ.global_type),
                [self.cols_field_name],
                lambda g: zip_with_index_field(g, col_uid_field_name))
            child = TableMapGlobals(child, new_globals)

        return CastTableToMatrix(child.handle_randomness(row_uid_field_name),
                                 self.entries_field_name,
                                 self.cols_field_name,
                                 self.col_key)

    def head_str(self):
        return '{} {} ({})'.format(
            escape_str(self.entries_field_name),
            escape_str(self.cols_field_name),
            ' '.join([escape_id(id) for id in self.col_key]))

    def _eq(self, other):
        return self.entries_field_name == other.entries_field_name and \
            self.cols_field_name == other.cols_field_name and \
            self.col_key == other.col_key

    def _compute_type(self, deep_typecheck):
        self.child.compute_type(deep_typecheck)
        child_typ = self.child.typ
        return hl.tmatrix(
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

    def _handle_randomness(self, row_uid_field_name, col_uid_field_name):
        return MatrixAnnotateRowsTable(
            self.child.handle_randomness(row_uid_field_name, col_uid_field_name),
            self.table.handle_randomness(None),
            self.root,
            self.product)

    def head_str(self):
        return f'"{escape_str(self.root)}" {self.product}'

    def _eq(self, other):
        return self.root == other.root and self.product == other.product

    def _compute_type(self, deep_typecheck):
        self.child.compute_type(deep_typecheck)
        self.table.compute_type(deep_typecheck)
        child_typ = self.child.typ
        if self.product:
            value_type = hl.tarray(self.table.typ.value_type)
        else:
            value_type = self.table.typ.value_type
        return hl.tmatrix(
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

    def _handle_randomness(self, row_uid_field_name, col_uid_field_name):
        return MatrixAnnotateColsTable(
            self.child.handle_randomness(row_uid_field_name, col_uid_field_name),
            self.table.handle_randomness(None),
            self.root)

    def head_str(self):
        return f'"{escape_str(self.root)}"'

    def _eq(self, other):
        return self.root == other.root

    def _compute_type(self, deep_typecheck):
        self.child.compute_type(deep_typecheck)
        self.table.compute_type(deep_typecheck)
        child_typ = self.child.typ
        return hl.tmatrix(
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

    def _handle_randomness(self, row_uid_field_name, col_uid_field_name):
        assert self.config['name'] == 'MatrixFilterPartitions'
        child = self.child.handle_randomness(row_uid_field_name, col_uid_field_name)
        return MatrixToMatrixApply(child, self.config)

    def head_str(self):
        return dump_json(self.config)

    def _eq(self, other):
        return self.config == other.config

    def _compute_type(self, deep_typecheck):
        self.child.compute_type(deep_typecheck)
        name = self.config['name']
        child_typ = self.child.typ
        assert name == 'MatrixFilterPartitions'
        return child_typ


class MatrixRename(MatrixIR):
    def __init__(self, child, global_map, col_map, row_map, entry_map):
        super().__init__(child)
        self.child = child
        self.global_map = global_map
        self.col_map = col_map
        self.row_map = row_map
        self.entry_map = entry_map

    def _handle_randomness(self, row_uid_field_name, col_uid_field_name):
        assert row_uid_field_name not in self.global_map.keys()
        assert row_uid_field_name not in self.col_map.keys()
        assert row_uid_field_name not in self.row_map.keys()
        assert row_uid_field_name not in self.entry_map.keys()
        assert col_uid_field_name not in self.global_map.keys()
        assert col_uid_field_name not in self.col_map.keys()
        assert col_uid_field_name not in self.row_map.keys()
        assert col_uid_field_name not in self.entry_map.keys()

        child = self.child.handle_randomness(row_uid_field_name, col_uid_field_name)
        return MatrixRename(child, self.global_map, self.col_map, self.row_map, self.entry_map)

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

    def _compute_type(self, deep_typecheck):
        self.child.compute_type(deep_typecheck)
        return self.child.typ._rename(self.global_map, self.col_map, self.row_map, self.entry_map)


class MatrixFilterIntervals(MatrixIR):
    def __init__(self, child, intervals, point_type, keep):
        super().__init__(child)
        self.child = child
        self.intervals = intervals
        self.point_type = point_type
        self.keep = keep

    def _handle_randomness(self, row_uid_field_name, col_uid_field_name):
        child = self.child.handle_randomness(row_uid_field_name, col_uid_field_name)
        return MatrixFilterIntervals(child, self.intervals, self.point_type, self.keep)

    def head_str(self):
        return f'{dump_json(hl.tarray(hl.tinterval(self.point_type))._convert_to_json(self.intervals))} {self.keep}'

    def _eq(self, other):
        return self.intervals == other.intervals and self.point_type == other.point_type and self.keep == other.keep

    def _compute_type(self, deep_typecheck):
        self.child.compute_type(deep_typecheck)
        return self.child.typ


class JavaMatrix(MatrixIR):
    def __init__(self, jir):
        super().__init__()
        self._jir = jir

    def _handle_randomness(self, row_uid_field_name, col_uid_field_name):
        raise FatalError('JavaMatrix does not support randomness in consumers')

    def render_head(self, r):
        return f'(JavaMatrix {r.add_jir(self._jir)}'

    def _compute_type(self, deep_typecheck):
        if self._type is None:
            return hl.tmatrix._from_java(self._jir.typ())
        else:
            return self._type

