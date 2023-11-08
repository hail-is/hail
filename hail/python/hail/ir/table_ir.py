from typing import Optional
import hail as hl
from hail.expr.types import dtype, tint32, tint64, tstruct
from hail.ir.base_ir import BaseIR, IR, TableIR
import hail.ir.ir as ir
from hail.ir.utils import modify_deep_field, zip_with_index, default_row_uid, default_col_uid
from hail.ir.ir import unify_uid_types, pad_uid, concat_uids
from hail.typecheck import typecheck_method, nullable, sequenceof
from hail.utils import FatalError
from hail.utils.interval import Interval
from hail.utils.java import Env
from hail.utils.misc import escape_str, parsable_strings, escape_id
from hail.utils.jsonx import dump_json


def unpack_uid(new_row_type, uid_field_name):
    new_row = ir.Ref('row', new_row_type)
    if uid_field_name in new_row_type.fields:
        uid = ir.GetField(new_row, uid_field_name)
    else:
        uid = ir.NA(tint64)
    return uid, \
        ir.SelectFields(new_row, [field for field in new_row_type.fields if not field == uid_field_name])


class MatrixRowsTable(TableIR):
    def __init__(self, child):
        super().__init__(child)
        self.child = child

    def _handle_randomness(self, uid_field_name):
        return MatrixRowsTable(self.child.handle_randomness(uid_field_name, None))

    def _compute_type(self, deep_typecheck):
        self.child.compute_type(deep_typecheck)
        return hl.ttable(self.child.typ.global_type,
                         self.child.typ.row_type,
                         self.child.typ.row_key)


class TableJoin(TableIR):
    def __init__(self, left, right, join_type, join_key):
        super().__init__(left, right)
        self.left = left
        self.right = right
        self.join_type = join_type
        self.join_key = join_key

    def _handle_randomness(self, uid_field_name):
        if uid_field_name is None:
            return TableJoin(self.left.handle_randomness(None),
                             self.right.handle_randomness(None),
                             self.join_type, self.join_key)

        left = self.left.handle_randomness('__left_uid')
        right = self.right.handle_randomness('__right_uid')

        joined = TableJoin(left, right, self.join_type, self.join_key)
        row = ir.Ref('row', joined.typ.row_type)
        old_joined_row = ir.SelectFields(row, [field for field in self.typ.row_type])
        left_uid = ir.GetField(row, '__left_uid')
        right_uid = ir.GetField(row, '__right_uid')
        handle_missing_left = self.join_type == 'right' or self.join_type == 'outer'
        handle_missing_right = self.join_type == 'left' or self.join_type == 'outer'
        uid = concat_uids(left_uid, right_uid, handle_missing_left, handle_missing_right)

        return TableMapRows(joined, ir.InsertFields(old_joined_row, [(uid_field_name, uid)], None))

    def head_str(self):
        return f'{escape_id(self.join_type)} {self.join_key}'

    def _eq(self, other):
        return self.join_key == other.join_key and \
            self.join_type == other.join_type

    def _compute_type(self, deep_typecheck):
        self.left.compute_type(deep_typecheck)
        self.right.compute_type(deep_typecheck)
        left_typ = self.left.typ
        right_typ = self.right.typ
        return hl.ttable(left_typ.global_type._concat(right_typ.global_type),
                         left_typ.key_type._concat(left_typ.value_type)._concat(right_typ.value_type),
                         left_typ.row_key + right_typ.row_key[self.join_key:])


class TableLeftJoinRightDistinct(TableIR):
    def __init__(self, left, right, root):
        super().__init__(left, right)
        self.left = left
        self.right = right
        self.root = root

    def _handle_randomness(self, uid_field_name):
        left = self.left.handle_randomness(uid_field_name)
        right = self.right.handle_randomness(None)
        return TableLeftJoinRightDistinct(left, right, self.root)

    def head_str(self):
        return escape_id(self.root)

    def _eq(self, other):
        return self.root == other.root

    def _compute_type(self, deep_typecheck):
        self.left.compute_type(deep_typecheck)
        self.right.compute_type(deep_typecheck)
        left_typ = self.left.typ
        right_typ = self.right.typ
        return hl.ttable(
            left_typ.global_type,
            left_typ.row_type._insert_field(self.root, right_typ.value_type),
            left_typ.row_key)


class TableIntervalJoin(TableIR):
    def __init__(self, left, right, root, product=False):
        super().__init__(left, right)
        self.left = left
        self.right = right
        self.root = root
        self.product = product

    def _handle_randomness(self, uid_field_name):
        left = self.left.handle_randomness(uid_field_name)
        right = self.right.handle_randomness(None)
        return TableIntervalJoin(left, right, self.root, self.product)

    def head_str(self):
        return f'{escape_id(self.root)} {self.product}'

    def _eq(self, other):
        return self.root == other.root and self.product == other.product

    def _compute_type(self, deep_typecheck):
        self.left.compute_type(deep_typecheck)
        self.right.compute_type(deep_typecheck)
        left_typ = self.left.typ
        right_typ = self.right.typ
        if self.product:
            right_val_typ = left_typ.row_type._insert_field(self.root, hl.tarray(right_typ.value_type))
        else:
            right_val_typ = left_typ.row_type._insert_field(self.root, right_typ.value_type)
        return hl.ttable(
            left_typ.global_type,
            right_val_typ,
            left_typ.row_key)


class TableUnion(TableIR):
    def __init__(self, children):
        super().__init__(*children)
        self.children = children

    def _handle_randomness(self, uid_field_name):
        if uid_field_name is None:
            new_children = [child.handle_randomness(None) for child in self.children]
            return TableUnion(new_children)

        new_children = [child.handle_randomness(uid_field_name) for child in self.children]

        if not all(uid_field_name in child.typ.row_type for child in new_children):
            new_children = [child.handle_randomness(None) for child in self.children]
            return TableUnion(new_children)

        uids = [uid for uid, _ in (unpack_uid(child.typ.row_type, uid_field_name) for child in new_children)]
        uid_type = unify_uid_types((uid.typ for uid in uids), tag=True)
        new_children = [TableMapRows(child,
                                     ir.InsertFields(ir.Ref('row', child.typ.row_type),
                                                     [(uid_field_name, pad_uid(uid, uid_type, i))], None))
                        for i, (child, uid) in enumerate(zip(new_children, uids))]
        return TableUnion(new_children)

    def _compute_type(self, deep_typecheck):
        for c in self.children:
            c.compute_type(deep_typecheck)
        return self.children[0].typ


class TableRange(TableIR):
    def __init__(self, n, n_partitions):
        super().__init__()
        self.n = n
        self.n_partitions = n_partitions

    def _handle_randomness(self, uid_field_name):
        assert(uid_field_name is not None)
        new_row = ir.InsertFields(ir.Ref('row', self.typ.row_type), [(uid_field_name, ir.Cast(ir.GetField(ir.Ref('row', self.typ.row_type), 'idx'), tint64))], None)
        return TableMapRows(self, new_row)

    def head_str(self):
        return f'{self.n} {self.n_partitions}'

    def _eq(self, other):
        return self.n == other.n and self.n_partitions == other.n_partitions

    def _compute_type(self, deep_typecheck):
        return hl.ttable(hl.tstruct(),
                         hl.tstruct(idx=hl.tint32),
                         ['idx'])


class TableMapGlobals(TableIR):
    def __init__(self, child, new_globals):
        super().__init__(child, new_globals)
        self.child = child
        self.new_globals = new_globals

    def _handle_randomness(self, uid_field_name):
        new_globals = self.new_globals
        if new_globals.uses_randomness:
            new_globals = ir.Let('__rng_state', ir.RNGStateLiteral(), new_globals)

        return TableMapGlobals(self.child.handle_randomness(uid_field_name),
                               new_globals)

    def _compute_type(self, deep_typecheck):
        self.new_globals.compute_type(self.child.typ.global_env(), None, deep_typecheck)
        return hl.ttable(self.new_globals.typ,
                         self.child.typ.row_type,
                         self.child.typ.row_key)

    def renderable_bindings(self, i, default_value=None):
        return self.child.typ.global_env(default_value) if i == 1 else {}


class TableExplode(TableIR):
    def __init__(self, child, path):
        super().__init__(child)
        self.child = child
        self.path = path

    def _handle_randomness(self, uid_field_name):
        if uid_field_name is None:
            return TableExplode(self.child.handle_randomness(None), self.path)

        child = self.child.handle_randomness(uid_field_name)

        new_row = modify_deep_field(ir.Ref('row', child.typ.row_type), self.path, zip_with_index)
        child = TableMapRows(child, new_row)

        new_explode = TableExplode(child, self.path)
        new_row = modify_deep_field(
            ir.Ref('row', new_explode.typ.row_type),
            self.path,
            lambda tuple: ir.GetTupleElement(tuple, 0),
            lambda row, tuple: ir.InsertFields(row, [(uid_field_name, concat_uids(ir.GetField(row, uid_field_name), ir.Cast(ir.GetTupleElement(tuple, 1), tint64)))], None))
        return TableMapRows(new_explode, new_row)

    def head_str(self):
        return parsable_strings(self.path)

    def _eq(self, other):
        return self.path == other.path

    def _compute_type(self, deep_typecheck):
        self.child.compute_type(deep_typecheck)
        atyp = self.child.typ.row_type._index_path(self.path)
        return hl.ttable(self.child.typ.global_type,
                         self.child.typ.row_type._insert(self.path, atyp.element_type),
                         self.child.typ.row_key)


class TableKeyBy(TableIR):
    def __init__(self, child, keys, is_sorted=False):
        super().__init__(child)
        self.child = child
        self.keys = keys
        self.is_sorted = is_sorted

    def _handle_randomness(self, uid_field_name):
        return TableKeyBy(self.child.handle_randomness(uid_field_name), self.keys, self.is_sorted)

    def head_str(self):
        return '({}) {}'.format(' '.join([escape_id(x) for x in self.keys]), self.is_sorted)

    def _eq(self, other):
        return self.keys == other.keys and self.is_sorted == other.is_sorted

    def _compute_type(self, deep_typecheck):
        self.child.compute_type(deep_typecheck)
        return hl.ttable(self.child.typ.global_type,
                         self.child.typ.row_type,
                         self.keys)


class TableMapRows(TableIR):
    def __init__(self, child, new_row):
        super().__init__(child, new_row)
        self.child = child
        self.new_row = new_row

    def _handle_randomness(self, uid_field_name):
        if not self.new_row.uses_randomness and uid_field_name is None:
            child = self.child.handle_randomness(None)
            return TableMapRows(child, self.new_row)

        child = self.child.handle_randomness(default_row_uid)
        uid, old_row = unpack_uid(child.typ.row_type, default_row_uid)
        new_row = ir.Let('row', old_row, self.new_row)
        if self.new_row.uses_value_randomness:
            new_row = ir.Let('__rng_state', ir.RNGSplit(ir.RNGStateLiteral(), uid), new_row)
        if self.new_row.uses_agg_randomness(is_scan=True):
            new_row = ir.AggLet('__rng_state', ir.RNGSplit(ir.RNGStateLiteral(), uid), new_row, is_scan=True)
        if uid_field_name is not None:
            new_row = ir.InsertFields(new_row, [(uid_field_name, uid)], None)
        return TableMapRows(child, new_row)

    def _compute_type(self, deep_typecheck):
        self.child.compute_type(deep_typecheck)
        # agg_env for scans
        self.new_row.compute_type(self.child.typ.row_env(), self.child.typ.row_env(), deep_typecheck)
        return hl.ttable(
            self.child.typ.global_type,
            self.new_row.typ,
            self.child.typ.row_key)

    def renderable_bindings(self, i, default_value=None):
        if i == 1:
            env = self.child.typ.row_env(default_value)
            env[BaseIR.agg_capability] = default_value
            return env
        else:
            return {}

    def renderable_scan_bindings(self, i, default_value=None):
        return self.child.typ.row_env(default_value) if i == 1 else {}


class TableMapPartitions(TableIR):
    def __init__(self, child, global_name, partition_stream_name, body, requested_key, allowed_overlap):
        super().__init__(child, body)
        self.child = child
        self.body = body
        self.global_name = global_name
        self.partition_stream_name = partition_stream_name
        self.requested_key = requested_key
        self.allowed_overlap = allowed_overlap

    def _handle_randomness(self, uid_field_name):
        if uid_field_name is not None:
            raise FatalError('TableMapPartitions does not support randomness, in its body or in consumers')
        return TableMapPartitions(self.child.handle_randomness(None), self.global_name, self.partition_stream_name, self.body, self.requested_key, self.allowed_overlap)

    def _compute_type(self, deep_typecheck):
        self.child.compute_type(deep_typecheck)
        self.body.compute_type({self.global_name: self.child.typ.global_type,
                                self.partition_stream_name: hl.tstream(self.child.typ.row_type)},
                               {},
                               deep_typecheck)
        assert isinstance(self.body.typ, hl.tstream) and isinstance(self.body.typ.element_type, hl.tstruct)
        new_row_type = self.body.typ.element_type
        for k in self.child.typ.row_key:
            assert k in new_row_type
        return hl.ttable(self.child.typ.global_type,
                         new_row_type,
                         self.child.typ.row_key)

    def renderable_bindings(self, i, default_value=None):
        if i == 1:
            return {self.global_name: self.child.typ.global_type if default_value is None else default_value,
                    self.partition_stream_name: hl.tstream(
                        self.child.typ.row_type) if default_value is None else default_value}
        else:
            return {}

    def head_str(self):
        return f'{escape_id(self.global_name)} {escape_id(self.partition_stream_name)} {self.requested_key} {self.allowed_overlap}'

    def _eq(self, other):
        return (self.global_name == other.global_name
                and self.partition_stream_name == other.partition_stream_name
                and self.allowed_overlap == other.allowed_overlap)


class TableRead(TableIR):
    def __init__(self,
                 reader,
                 drop_rows: bool = False,
                 drop_row_uids: bool = True,
                 *,
                 _assert_type: Optional['hl.ttable'] = None):
        super().__init__()
        self.reader = reader
        self.drop_rows = drop_rows
        self.drop_row_uids = drop_row_uids
        self._type = _assert_type

    def _handle_randomness(self, uid_field_name):
        rename_row_uid = False
        drop_row_uids = False
        if uid_field_name is None and self.drop_row_uids:
            drop_row_uids = True
        elif uid_field_name is not None and uid_field_name != default_row_uid:
            rename_row_uid = True
        result = TableRead(self.reader, self.drop_rows, drop_row_uids)
        if rename_row_uid:
            if self.drop_row_uids:
                result = TableRename(result, {default_row_uid: uid_field_name}, {})
            else:
                row = ir.Ref('row', self.typ.row_type)
                result = TableMapRows(
                    result,
                    ir.InsertFields(row, [(uid_field_name, ir.GetField(row, default_row_uid))], None))
        return result

    def head_str(self):
        if self.drop_row_uids:
            reqType = "DropRowUIDs"
        else:
            reqType = "None"
        return f'{reqType} {self.drop_rows} "{self.reader.render()}"'

    def _eq(self, other):
        return (self.reader == other.reader
                and self.drop_rows == other.drop_rows
                and self.drop_row_uids == other.drop_row_uids)

    def _compute_type(self, deep_typecheck):
        if self._type is not None:
            return self._type
        else:
            return Env.backend().table_type(self)


class MatrixEntriesTable(TableIR):
    def __init__(self, child):
        super().__init__(child)
        self.child = child

    def _handle_randomness(self, uid_field_name):
        from hail.ir.matrix_ir import MatrixMapEntries, MatrixMapRows
        if uid_field_name is None:
            return MatrixEntriesTable(self.child.handle_randomness(None, None))

        temp_row_uid = Env.get_uid(default_row_uid)
        child = self.child.handle_randomness(temp_row_uid, default_col_uid)
        entry = ir.Ref('g', child.typ.entry_type)
        row_uid = ir.GetField(ir.Ref('va', child.typ.row_type), temp_row_uid)
        col_uid = ir.GetField(ir.Ref('sa', child.typ.col_type), default_col_uid)
        child = MatrixMapEntries(child, ir.InsertFields(entry, [('__entry_uid', ir.concat_uids(row_uid, col_uid))], None))
        child = MatrixMapRows(child, ir.SelectFields(ir.Ref('va', child.typ.row_type), [field for field in child.typ.row_type if field != temp_row_uid]))
        return TableRename(MatrixEntriesTable(child), {'__entry_uid': default_row_uid}, {})

    def _compute_type(self, deep_typecheck):
        self.child.compute_type(deep_typecheck)
        child_typ = self.child.typ
        return hl.ttable(child_typ.global_type,
                         child_typ.row_type
                                  ._concat(child_typ.col_type)
                                  ._concat(child_typ.entry_type),
                         child_typ.row_key + child_typ.col_key)


class TableFilter(TableIR):
    def __init__(self, child, pred):
        super().__init__(child, pred)
        self.child = child
        self.pred = pred

    def _handle_randomness(self, uid_field_name):
        if not self.pred.uses_randomness and uid_field_name is None:
            child = self.child.handle_randomness(None)
            return TableFilter(child, self.pred)

        drop_uid = uid_field_name is None
        if uid_field_name is None:
            uid_field_name = ir.uid_field_name
        child = self.child.handle_randomness(uid_field_name)
        uid, old_row = unpack_uid(child.typ.row_type, uid_field_name)
        pred = ir.Let('row', old_row, self.pred)
        if self.pred.uses_randomness:
            pred = ir.Let('__rng_state', ir.RNGSplit(ir.RNGStateLiteral(), uid), pred)
        result = TableFilter(child, pred)
        if drop_uid:
            result = TableMapRows(result, old_row)
        return result

    def _compute_type(self, deep_typecheck):
        self.child.compute_type(deep_typecheck)
        self.pred.compute_type(self.child.typ.row_env(), None, deep_typecheck)
        return self.child.typ

    def renderable_bindings(self, i, default_value=None):
        return self.child.typ.row_env(default_value) if i == 1 else {}


class TableKeyByAndAggregate(TableIR):
    def __init__(self, child, expr, new_key, n_partitions, buffer_size):
        super().__init__(child, expr, new_key)
        self.child = child
        self.expr = expr
        self.new_key = new_key
        self.n_partitions = n_partitions
        self.buffer_size = buffer_size

    def _handle_randomness(self, uid_field_name):
        if not self.expr.uses_randomness and not self.new_key.uses_randomness and uid_field_name is None:
            child = self.child.handle_randomness(None)
            return TableKeyByAndAggregate(child, self.expr, self.new_key, self.n_partitions, self.buffer_size)

        child = self.child.handle_randomness(ir.uid_field_name)
        uid, old_row = unpack_uid(child.typ.row_type, ir.uid_field_name)

        expr = self.expr
        if expr.uses_randomness or uid_field_name is not None:
            first_uid = ir.Ref(Env.get_uid(), uid.typ)
            if expr.uses_randomness:
                expr = ir.Let(
                    '__rng_state',
                    ir.RNGSplit(ir.RNGStateLiteral(), first_uid),
                    expr)
            if expr.uses_agg_randomness(is_scan=False):
                expr = ir.AggLet('__rng_state',
                                 ir.RNGSplit(ir.RNGStateLiteral(), uid),
                                 expr, is_scan=False)
            if uid_field_name is not None:
                expr = ir.InsertFields(expr, [(uid_field_name, first_uid)], None)
            expr = ir.Let(first_uid.name, ir.ArrayRef(ir.ApplyAggOp('Take', [ir.I32(1)], [uid]), ir.I32(0)), expr)
        new_key = self.new_key
        if new_key.uses_randomness:
            expr = ir.Let(
                '__rng_state',
                ir.RNGSplit(ir.RNGStateLiteral(), uid),
                new_key)
        return TableKeyByAndAggregate(child, expr, new_key, self.n_partitions, self.buffer_size)

    def head_str(self):
        return f'{self.n_partitions} {self.buffer_size}'

    def _eq(self, other):
        return self.n_partitions == other.n_partitions and self.buffer_size == other.buffer_size

    def _compute_type(self, deep_typecheck):
        self.child.compute_type(deep_typecheck)
        self.expr.compute_type(self.child.typ.global_env(), self.child.typ.row_env(), deep_typecheck)
        self.new_key.compute_type(self.child.typ.row_env(), None, deep_typecheck)
        return hl.ttable(self.child.typ.global_type,
                         self.new_key.typ._concat(self.expr.typ),
                         list(self.new_key.typ))

    def renderable_bindings(self, i, default_value=None):
        if i == 1:
            env = self.child.typ.global_env(default_value)
            env[BaseIR.agg_capability] = default_value
            return env
        elif i == 2:
            return self.child.typ.row_env(default_value)
        else:
            return {}

    def renderable_agg_bindings(self, i, default_value=None):
        return self.child.typ.row_env(default_value) if i == 1 else {}


class TableAggregateByKey(TableIR):
    def __init__(self, child, expr):
        super().__init__(child, expr)
        self.child = child
        self.expr = expr

    def _handle_randomness(self, uid_field_name):
        if not self.expr.uses_randomness and uid_field_name is None:
            child = self.child.handle_randomness(None)
            return TableAggregateByKey(child, self.expr)

        child = self.child.handle_randomness(ir.uid_field_name)
        uid, old_row = unpack_uid(child.typ.row_type, ir.uid_field_name)

        expr = ir.AggLet('va', old_row, self.expr, is_scan=False)
        first_uid = ir.Ref(Env.get_uid(), uid.typ)
        if expr.uses_value_randomness:
            expr = ir.Let(
                '__rng_state',
                ir.RNGSplit(ir.RNGStateLiteral(), first_uid),
                expr)
        if expr.uses_agg_randomness(is_scan=False):
            expr = ir.AggLet(
                '__rng_state',
                ir.RNGSplit(ir.RNGStateLiteral(), uid),
                expr,
                is_scan=False)
        if uid_field_name is not None:
            expr = ir.InsertFields(expr, [(uid_field_name, first_uid)], None)
        expr = ir.Let(first_uid.name, ir.ArrayRef(ir.ApplyAggOp('Take', [ir.I32(1)], [uid]), ir.I32(0)), expr)
        return TableAggregateByKey(child, expr)

    def _compute_type(self, deep_typecheck):
        self.child.compute_type(deep_typecheck)
        child_typ = self.child.typ
        self.expr.compute_type(child_typ.global_env(), child_typ.row_env(), deep_typecheck)
        return hl.ttable(child_typ.global_type,
                         child_typ.key_type._concat(self.expr.typ),
                         child_typ.row_key)

    def renderable_bindings(self, i, default_value=None):
        if i == 1:
            env = self.child.typ.row_env(default_value)
            env[BaseIR.agg_capability] = default_value
            return env
        else:
            return {}

    def renderable_agg_bindings(self, i, default_value=None):
        return self.child.typ.row_env(default_value) if i == 1 else {}


class MatrixColsTable(TableIR):
    def __init__(self, child):
        super().__init__(child)
        self.child = child

    def _handle_randomness(self, uid_field_name):
        return MatrixColsTable(self.child.handle_randomness(None, uid_field_name))

    def _compute_type(self, deep_typecheck):
        self.child.compute_type(deep_typecheck)
        return hl.ttable(self.child.typ.global_type,
                         self.child.typ.col_type,
                         self.child.typ.col_key)


class TableParallelize(TableIR):
    def __init__(self, rows_and_global, n_partitions):
        super().__init__(rows_and_global)
        self.rows_and_global = rows_and_global
        self.n_partitions = n_partitions

    def _handle_randomness(self, uid_field_name):
        rows_and_global = self.rows_and_global
        if rows_and_global.uses_randomness:
            rows_and_global = ir.Let(
                '__rng_state',
                ir.RNGStateLiteral(),
                rows_and_global)
        if uid_field_name is not None:
            rows_and_global_ref = ir.Ref(Env.get_uid(), rows_and_global.typ)
            row = Env.get_uid()
            uid = Env.get_uid()
            iota = ir.StreamIota(ir.I32(0), ir.I32(1))
            rows = ir.ToStream(ir.GetField(rows_and_global_ref, 'rows'))
            new_rows = ir.ToArray(ir.StreamZip(
                [rows, iota],
                [row, uid],
                ir.InsertFields(
                    ir.Ref(row, rows.typ.element_type),
                    [(uid_field_name, ir.Cast(ir.Ref(uid, tint32), tint64))],
                    None),
                'TakeMinLength'))
            rows_and_global = \
                ir.Let(rows_and_global_ref.name, rows_and_global,
                       ir.InsertFields(
                           rows_and_global_ref,
                           [('rows', new_rows)], None))
        return TableParallelize(rows_and_global, self.n_partitions)

    def head_str(self):
        return self.n_partitions

    def _eq(self, other):
        return self.n_partitions == other.n_partitions

    def _compute_type(self, deep_typecheck):
        self.rows_and_global.compute_type({}, None, deep_typecheck)
        return hl.ttable(self.rows_and_global.typ['global'],
                         self.rows_and_global.typ['rows'].element_type,
                         [])


class TableHead(TableIR):
    def __init__(self, child, n):
        super().__init__(child)
        self.child = child
        self.n = n

    def _handle_randomness(self, uid_field_name):
        return TableHead(self.child.handle_randomness(uid_field_name), self.n)

    def head_str(self):
        return self.n

    def _eq(self, other):
        return self.n == other.n

    def _compute_type(self, deep_typecheck):
        self.child.compute_type(deep_typecheck)
        return self.child.typ


class TableTail(TableIR):
    def __init__(self, child, n):
        super().__init__(child)
        self.child = child
        self.n = n

    def _handle_randomness(self, uid_field_name):
        return TableTail(self.child.handle_randomness(uid_field_name), self.n)

    def head_str(self):
        return self.n

    def _eq(self, other):
        return self.n == other.n

    def _compute_type(self, deep_typecheck):
        self.child.compute_type(deep_typecheck)
        return self.child.typ


class TableOrderBy(TableIR):
    def __init__(self, child, sort_fields):
        super().__init__(child)
        self.child = child
        self.sort_fields = sort_fields

    def _handle_randomness(self, uid_field_name):
        return TableOrderBy(self.child.handle_randomness(uid_field_name), self.sort_fields)

    def head_str(self):
        return f'({" ".join([escape_id(order + f) for (f, order) in self.sort_fields])})'

    def _eq(self, other):
        return self.sort_fields == other.sort_fields

    def _compute_type(self, deep_typecheck):
        self.child.compute_type(deep_typecheck)
        return hl.ttable(self.child.typ.global_type,
                         self.child.typ.row_type,
                         [])


class TableDistinct(TableIR):
    def __init__(self, child):
        super().__init__(child)
        self.child = child

    def _handle_randomness(self, uid_field_name):
        return TableDistinct(self.child.handle_randomness(uid_field_name))

    def _compute_type(self, deep_typecheck):
        self.child.compute_type(deep_typecheck)
        return self.child.typ


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

    def _handle_randomness(self, uid_field_name):
        return TableRepartition(self.child.handle_randomness(uid_field_name), self.n, self.strategy)

    def head_str(self):
        return f'{self.n} {self.strategy}'

    def _eq(self, other):
        return self.n == other.n and self.strategy == other.strategy

    def _compute_type(self, deep_typecheck):
        self.child.compute_type(deep_typecheck)
        return self.child.typ


class CastMatrixToTable(TableIR):
    def __init__(self, child, entries_field_name, cols_field_name):
        super().__init__(child)
        self.child = child
        self.entries_field_name = entries_field_name
        self.cols_field_name = cols_field_name

    def _handle_randomness(self, uid_field_name):
        return CastMatrixToTable(self.child.handle_randomness(uid_field_name, None), self.entries_field_name, self.cols_field_name)

    def head_str(self):
        return f'"{escape_str(self.entries_field_name)}" "{escape_str(self.cols_field_name)}"'

    def _eq(self, other):
        return self.entries_field_name == other.entries_field_name and self.cols_field_name == other.cols_field_name

    def _compute_type(self, deep_typecheck):
        self.child.compute_type(deep_typecheck)
        child_typ = self.child.typ
        return hl.ttable(child_typ.global_type._insert_field(self.cols_field_name, hl.tarray(child_typ.col_type)),
                         child_typ.row_type._insert_field(self.entries_field_name,
                                                          hl.tarray(child_typ.entry_type)),
                         child_typ.row_key)


class TableRename(TableIR):
    def __init__(self, child, row_map, global_map):
        super().__init__(child)
        self.child = child
        self.row_map = row_map
        self.global_map = global_map

    def _handle_randomness(self, uid_field_name):
        return TableRename(self.child.handle_randomness(uid_field_name), self.row_map, self.global_map)

    def head_str(self):
        return f'{parsable_strings(self.row_map.keys())} ' \
               f'{parsable_strings(self.row_map.values())} ' \
               f'{parsable_strings(self.global_map.keys())} ' \
               f'{parsable_strings(self.global_map.values())} '

    def _eq(self, other):
        return self.row_map == other.row_map and self.global_map == other.global_map

    def _compute_type(self, deep_typecheck):
        self.child.compute_type(deep_typecheck)
        return self.child.typ._rename(self.global_map, self.row_map)


class TableMultiWayZipJoin(TableIR):
    def __init__(self, children, data_name, global_name):
        super().__init__(*children)
        self.children = children
        self.data_name = data_name
        self.global_name = global_name

    def _handle_randomness(self, uid_field_name):
        if uid_field_name is None:
            new_children = [child.handle_randomness(None) for child in self.children]
            return TableMultiWayZipJoin(new_children, self.data_name, self.global_name)

        new_children = [child.handle_randomness(uid_field_name) for child in self.children]
        uids = [uid for uid, _ in (unpack_uid(child.typ.row_type, uid_field_name) for child in new_children)]
        uid_type = unify_uid_types((uid.typ for uid in uids), tag=True)
        new_children = [
            TableMapRows(
                child,
                ir.InsertFields(ir.Ref('row', child.typ.row_type),
                                [(uid_field_name,
                                  pad_uid(uid, uid_type, i))], None))
            for i, (child, uid) in enumerate(zip(new_children, uids))]
        joined = TableMultiWayZipJoin(new_children, self.data_name, self.global_name)
        accum = ir.Ref(Env.get_uid(), uid_type)
        elt = Env.get_uid()
        row = ir.Ref('row', joined.typ.row_type)
        data = ir.GetField(row, self.data_name)
        uid = ir.StreamFold(
            ir.toStream(data), ir.NA(uid_type), accum.name, elt,
            ir.If(ir.IsNA(accum),
                  ir.GetField(
                      ir.Ref(elt, data.typ.element_type),
                      uid_field_name),
                  accum))
        return TableMapRows(joined, ir.InsertFields(row, [(uid_field_name, uid)], None))

    def head_str(self):
        return f'"{escape_str(self.data_name)}" "{escape_str(self.global_name)}"'

    def _eq(self, other):
        return self.data_name == other.data_name and self.global_name == other.global_name

    def _compute_type(self, deep_typecheck):
        for c in self.children:
            c.compute_type(deep_typecheck)
        child_typ = self.children[0].typ
        return hl.ttable(
            hl.tstruct(**{self.global_name: hl.tarray(child_typ.global_type)}),
            child_typ.key_type._insert_field(self.data_name, hl.tarray(child_typ.value_type)),
            child_typ.row_key)


class TableFilterIntervals(TableIR):
    def __init__(self, child, intervals, point_type, keep):
        super().__init__(child)
        self.child = child
        self.intervals = intervals
        self.point_type = point_type
        self.keep = keep

    def _handle_randomness(self, uid_field_name):
        return TableFilterIntervals(self.child.handle_randomness(uid_field_name), self.intervals, self.point_type, self.keep)

    def head_str(self):
        return f'{self.child.typ.key_type()._parsable_string()} {dump_json(hl.tarray(hl.tinterval(self.point_type))._convert_to_json(self.intervals))} {self.keep}'

    def _eq(self, other):
        return self.intervals == other.intervals and self.point_type == other.point_type and self.keep == other.keep

    def _compute_type(self, deep_typecheck):
        self.child.compute_type(deep_typecheck)
        return self.child.typ


class TableToTableApply(TableIR):
    def __init__(self, child, config):
        super().__init__(child)
        self.child = child
        self.config = config

    def _handle_randomness(self, uid_field_name):
        if uid_field_name is not None:
            if self.config['name'] != 'TableFilterPartitions':
                raise FatalError(f'TableToTableApply({self.config["name"]}) does not support randomness in consumers')
        child = self.child.handle_randomness(uid_field_name)
        return TableToTableApply(child, self.config)

    def head_str(self):
        return dump_json(self.config)

    def _eq(self, other):
        return self.config == other.config

    def _compute_type(self, deep_typecheck):
        self.child.compute_type(deep_typecheck)
        name = self.config['name']
        if name == 'TableFilterPartitions':
            return self.child.typ
        else:
            assert name in ('VEP', 'Nirvana'), name
            if self._type is not None:
                return self._type
            else:
                return Env.backend().table_type(self)


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

    def _handle_randomness(self, uid_field_name):
        if uid_field_name is not None:
            raise FatalError('TableToTableApply does not support randomness in consumers')
        child = self.child.handle_randomness(None, None)
        return MatrixToTableApply(child, self.config)

    def head_str(self):
        return dump_json(self.config)

    def _eq(self, other):
        return self.config == other.config

    def _compute_type(self, deep_typecheck):
        self.child.compute_type(deep_typecheck)
        name = self.config['name']
        child_typ = self.child.typ
        if name == 'LinearRegressionRowsChained':
            pass_through = self.config['passThrough']
            chained_schema = hl.dtype(
                'struct{n:array<int32>,sum_x:array<float64>,y_transpose_x:array<array<float64>>,beta:array<array<float64>>,standard_error:array<array<float64>>,t_stat:array<array<float64>>,p_value:array<array<float64>>}')
            return hl.ttable(
                child_typ.global_type,
                (child_typ.row_key_type
                 ._insert_fields(**{f: child_typ.row_type[f] for f in pass_through})
                 ._concat(chained_schema)),
                child_typ.row_key)
        elif name == 'LinearRegressionRowsSingle':
            pass_through = self.config['passThrough']
            chained_schema = hl.dtype(
                'struct{n:int32,sum_x:float64,y_transpose_x:array<float64>,beta:array<float64>,standard_error:array<float64>,t_stat:array<float64>,p_value:array<float64>}')
            return hl.ttable(
                child_typ.global_type,
                (child_typ.row_key_type
                 ._insert_fields(**{f: child_typ.row_type[f] for f in pass_through})
                 ._concat(chained_schema)),
                child_typ.row_key)
        elif name == 'LogisticRegression':
            pass_through = self.config['passThrough']
            logreg_type = hl.tstruct(logistic_regression=hl.tarray(regression_test_type(self.config['test'])))
            return hl.ttable(
                child_typ.global_type,
                (child_typ.row_key_type
                 ._insert_fields(**{f: child_typ.row_type[f] for f in pass_through})
                 ._concat(logreg_type)),
                child_typ.row_key)
        elif name == 'PoissonRegression':
            pass_through = self.config['passThrough']
            poisreg_type = regression_test_type(self.config['test'])
            return hl.ttable(
                child_typ.global_type,
                (child_typ.row_key_type
                 ._insert_fields(**{f: child_typ.row_type[f] for f in pass_through})
                 ._concat(poisreg_type)),
                child_typ.row_key)
        elif name == 'Skat':
            key_field = self.config['keyField']
            key_type = child_typ.row_type[key_field]
            skat_type = hl.dtype(f'struct{{id:{key_type},size:int32,q_stat:float64,p_value:float64,fault:int32}}')
            return hl.ttable(
                hl.tstruct(),
                skat_type,
                ['id'])
        elif name == 'PCA':
            return hl.ttable(
                hl.tstruct(eigenvalues=hl.tarray(hl.tfloat64),
                           scores=hl.tarray(child_typ.col_key_type._insert_field('scores', hl.tarray(hl.tfloat64)))),
                child_typ.row_key_type._insert_field('loadings', dtype('array<float64>')),
                child_typ.row_key)
        elif name == 'IBD':
            ibd_info_type = hl.tstruct(Z0=hl.tfloat64, Z1=hl.tfloat64, Z2=hl.tfloat64, PI_HAT=hl.tfloat64)
            ibd_type = hl.tstruct(i=hl.tstr,
                                  j=hl.tstr,
                                  ibd=ibd_info_type,
                                  ibs0=hl.tint64,
                                  ibs1=hl.tint64,
                                  ibs2=hl.tint64)
            return hl.ttable(
                hl.tstruct(),
                ibd_type,
                ['i', 'j'])
        else:
            assert name == 'LocalLDPrune', name
            return hl.ttable(
                hl.tstruct(),
                child_typ.row_key_type._insert_fields(mean=hl.tfloat64, centered_length_rec=hl.tfloat64),
                list(child_typ.row_key))


class BlockMatrixToTableApply(TableIR):
    def __init__(self, bm, aux, config):
        super().__init__(bm, aux)
        self.bm = bm
        self.aux = aux
        self.config = config

    def _handle_randomness(self, uid_field_name):
        raise FatalError('BlockMatrixToTableApply does not support randomness in consumers')

    def head_str(self):
        return dump_json(self.config)

    def _eq(self, other):
        return self.config == other.config

    def _compute_type(self, deep_typecheck):
        self.bm.compute_type(deep_typecheck)
        self.aux.compute_type({}, None, deep_typecheck)
        name = self.config['name']
        assert name == 'PCRelate', name
        return hl.ttable(
            hl.tstruct(),
            hl.tstruct(i=hl.tint32, j=hl.tint32,
                       kin=hl.tfloat64,
                       ibd0=hl.tfloat64,
                       ibd1=hl.tfloat64,
                       ibd2=hl.tfloat64),
            ['i', 'j'])


class BlockMatrixToTable(TableIR):
    def __init__(self, child):
        super().__init__(child)
        self.child = child

    def _handle_randomness(self, uid_field_name):
        result = self
        if uid_field_name is not None:
            row = ir.Ref('row', result.typ.row_type)
            new_row = ir.InsertFields(row, [(uid_field_name,
                                             ir.MakeTuple([ir.GetField(row, 'i'), ir.GetField(row, 'j')]))], None)
            result = TableMapRows(result, new_row)
        return result

    def _compute_type(self, deep_typecheck):
        self.child.compute_type(deep_typecheck)
        return hl.ttable(hl.tstruct(), hl.tstruct(**{'i': hl.tint64, 'j': hl.tint64, 'entry': hl.tfloat64}), [])


class Partitioner(object):
    @typecheck_method(
        key_type=tstruct,
        range_bounds=sequenceof(Interval)
    )
    def __init__(self, key_type, range_bounds):
        assert all(map(lambda interval: interval.point_type == key_type, range_bounds))
        self._key_type = key_type
        self._range_bounds = range_bounds
        self._serialized_type = hl.tarray(hl.tinterval(key_type))

    @property
    def key_type(self):
        return self._key_type

    @property
    def range_bounds(self):
        return self._range_bounds

    def _parsable_string(self):
        return (
            f'Partitioner {self.key_type._parsable_string()} ' + dump_json(
                self._serialized_type._convert_to_json(self.range_bounds)
            )
        )

    def __str__(self):
        return f'Partitioner<{self.key_type}> {self.range_bounds}'


class TableGen(TableIR):
    @typecheck_method(
        contexts=IR,
        globals=IR,
        cname=str,
        gname=str,
        body=IR,
        partitioner=Partitioner,
        error_id=nullable(int)
    )
    def __init__(self, contexts, globals, cname, gname, body, partitioner, error_id=None):
        super().__init__(contexts, globals, body)
        self.contexts = contexts
        self.globals = globals
        self.cname = cname
        self.gname = gname
        self.body = body
        self.partitioner = partitioner
        self._error_id = error_id
        if error_id is None:
            self.save_error_info()

    def _compute_type(self, deep_typecheck):
        self.contexts.compute_type({}, None, deep_typecheck)
        self.globals.compute_type({}, None, deep_typecheck)
        bodyenv = {
            self.cname: self.contexts.typ.element_type,
            self.gname: self.globals.typ
        }
        self.body.compute_type(bodyenv, None, deep_typecheck)
        return hl.ttable(
            global_type=self.globals.typ,
            row_type=self.body.typ.element_type,
            row_key=self.partitioner.key_type.fields
        )

    def renderable_bindings(self, i, default_value=None):
        return {} if i != 2 else {
            self.cname: self.contexts.type.element_type if default_value is None else default_value,
            self.gname: self.globals.type if default_value is None else default_value
        }

    def _eq(self, other):
        return (
            self.cname == other.cname
            and self.gname == other.gname
            and self.partitioner == other.partitioner
            and self._error_id == other._error_id
        )

    def head_str(self):
        return ' '.join([
            self.cname,
            self.gname,
            '(' + self.partitioner._parsable_string() + ')',
            str(self._error_id)
        ])

    def _handle_randomness(self, uid_field_name):
        globals = self.globals
        if globals.uses_randomness:
            globals = ir.Let('__rng_state', ir.RNGStateLiteral(), globals)

        contexts = self.contexts
        if contexts.uses_randomness:
            contexts = ir.Let('__rng_state', ir.RNGStateLiteral(), contexts)

        contexts = contexts.handle_randomness(create_uids=True)
        cname, random_uid, old_context = ir.unpack_uid(contexts.typ)
        body = ir.Let(self.cname, old_context, self.body)

        if body.uses_randomness:
            body = ir.Let('__rng_state', ir.RNGStateLiteral(),
                          ir.with_split_rng_state(body, random_uid))

        if uid_field_name is not None:
            idx = ir.Ref(Env.get_uid(), ir.tint32)
            elem = ir.Ref(Env.get_uid(), body.typ.element_type)
            insert = ir.InsertFields(
                elem,
                [(uid_field_name, concat_uids(random_uid, ir.Cast(idx, ir.tint64)))],
                None
            )

            iota = ir.StreamIota(ir.I32(0), ir.I32(1))
            body = ir.StreamZip([iota, body], [idx.name, elem.name], insert, 'TakeMinLength')

        return TableGen(
            contexts,
            globals,
            cname,
            self.gname,
            body,
            self.partitioner,
            self._error_id
        )


class JavaTable(TableIR):
    def __init__(self, table_type, tir_id: int):
        super().__init__()
        self._type = table_type
        self._id = tir_id

    def _handle_randomness(self, uid_field_name):
        raise FatalError('JavaTable does not support randomness in consumers')

    def render_head(self, r):
        return f'(JavaTable {self._id}'

    def _compute_type(self, deep_typecheck):
        return self._type

    def __del__(self):
        from hail.backend.py4j_backend import Py4JBackend
        if Env._hc:
            backend = Env.backend()
            assert isinstance(backend, Py4JBackend)
            backend._jbackend.removeJavaIR(self._id)
