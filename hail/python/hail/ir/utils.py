from typing import Any, List, Optional, Tuple

import hail as hl
from hail.expr.types import tint32, tint64
from hail.utils.java import Env


def finalize_randomness(x):
    from hail.ir import ir

    if isinstance(x, ir.IR):
        x = ir.Let('__rng_state', ir.RNGStateLiteral(), x)
    elif isinstance(x, ir.TableIR):
        x = x.handle_randomness(None)
    elif isinstance(x, ir.MatrixIR):
        x = x.handle_randomness(None, None)
    return x


default_row_uid = '__row_uid'
default_col_uid = '__col_uid'


def unpack_row_uid(new_row_type, uid_field_name, drop_uid=True):
    from hail.ir import ir

    new_row = ir.Ref('va', new_row_type)
    if uid_field_name in new_row_type.fields:
        uid = ir.GetField(new_row, uid_field_name)
    else:
        uid = ir.NA(tint64)
    if drop_uid:
        new_row = ir.SelectFields(new_row, [field for field in new_row_type.fields if not field == uid_field_name])
    return uid, new_row


def unpack_col_uid(new_col_type, uid_field_name):
    from hail.ir import ir

    new_row = ir.Ref('sa', new_col_type)
    if uid_field_name in new_col_type.fields:
        uid = ir.GetField(new_row, uid_field_name)
    else:
        uid = ir.NA(tint64)
    return uid, ir.SelectFields(new_row, [field for field in new_col_type.fields if not field == uid_field_name])


def modify_deep_field(struct, path, new_deep_field, new_struct=None):
    from hail.ir import ir

    refs = [struct]
    for i in range(len(path)):
        refs.append(ir.Ref(Env.get_uid(), refs[i].typ[path[i]]))

    acc = new_deep_field(refs[-1])
    for parent_struct, field_name in zip(refs[-2::-1], path[::-1]):
        acc = ir.InsertFields(parent_struct, [(field_name, acc)], None)
    if new_struct is not None:
        acc = new_struct(acc, refs[-1])
    for struct_ref, field_ref, field_name in zip(refs[-2::-1], refs[:0:-1], path[::-1]):
        acc = ir.Let(field_ref.name, ir.GetField(struct_ref, field_name), acc)
    return acc


def zip_with_index(array):
    from hail.ir import ir

    elt = Env.get_uid()
    inner_row_uid = Env.get_uid()
    iota = ir.StreamIota(ir.I32(0), ir.I32(1))
    return ir.toArray(
        ir.StreamZip(
            [ir.toStream(array), iota],
            [elt, inner_row_uid],
            ir.MakeTuple((ir.Ref(elt, array.typ.element_type), ir.Ref(inner_row_uid, tint32))),
            'TakeMinLength',
        )
    )


def zip_with_index_field(array, idx_field_name):
    from hail.ir import ir

    elt = Env.get_uid()
    inner_row_uid = Env.get_uid()
    iota = ir.StreamIota(ir.I32(0), ir.I32(1))
    return ir.toArray(
        ir.StreamZip(
            [ir.toStream(array), iota],
            [elt, inner_row_uid],
            ir.InsertFields(
                ir.Ref(elt, array.typ.element_type),
                [(idx_field_name, ir.Cast(ir.Ref(inner_row_uid, tint32), tint64))],
                None,
            ),
            'TakeMinLength',
        )
    )


def impute_type_of_partition_interval_array(intervals: Optional[List[Any]]) -> Tuple[Optional[List[Any]], Any]:
    if intervals is None:
        return None, None
    if len(intervals) == 0:
        return [], hl.tarray(hl.tinterval(hl.tstruct()))

    t = hl.expr.impute_type(intervals)
    if not isinstance(t, hl.tarray) or not isinstance(t.element_type, hl.tinterval):
        raise TypeError("'intervals' must be an array of tintervals")
    pt = t.element_type.point_type

    if isinstance(pt, hl.tstruct):
        return intervals, t

    struct_intervals = [
        hl.Interval(hl.Struct(__point=i.start), hl.Struct(__point=i.end), i.includes_start, i.includes_end)
        for i in intervals
    ]
    struct_intervals_type = hl.tarray(hl.tinterval(hl.tstruct(__point=pt)))
    return struct_intervals, struct_intervals_type


def filter_predicate_with_keep(ir_pred, keep):
    from hail.ir import ir

    return ir.Coalesce(ir_pred if keep else ir.ApplyUnaryPrimOp('!', ir_pred), ir.FalseIR())


def make_filter_and_replace(filter, find_replace):
    if find_replace is None:
        find = None
        replace = None
    else:
        find, replace = find_replace
    return {'filterPattern': filter, 'findPattern': find, 'replacePattern': replace}


def parse_type(string_expr, ttype):
    if ttype == hl.tstr:
        return string_expr
    elif ttype == hl.tint32:
        return hl.int32(string_expr)
    elif ttype == hl.tint64:
        return hl.int64(string_expr)
    elif ttype == hl.tfloat32:
        return hl.float32(string_expr)
    elif ttype == hl.tfloat64:
        return hl.float64(string_expr)
    elif ttype == hl.tbool:
        return hl.bool(string_expr)
    elif ttype == hl.tcall:
        return hl.parse_call(string_expr)
    elif isinstance(ttype, hl.tlocus):
        return hl.parse_locus(string_expr, ttype.reference_genome)
    elif isinstance(ttype, hl.tinterval) and isinstance(ttype.point_type, hl.tlocus):
        return hl.parse_locus_interval(string_expr, ttype.point_type.reference_genome)
    else:
        return hl.parse_json(string_expr, ttype)
