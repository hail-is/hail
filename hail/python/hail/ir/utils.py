from typing import Optional, List, Any, Tuple
import hail as hl
from .ir import Let, Ref, GetField, InsertFields, RNGStateLiteral, Coalesce, ApplyUnaryPrimOp, FalseIR, BaseIR, IR, StreamIota, StreamZip, ToStream, I32, MakeTuple
from hail.utils.java import Env


def finalize_randomness(ir, key=(0, 0, 0, 0)) -> 'BaseIR':
    if isinstance(ir, IR):
        ir = Let('__rng_state', RNGStateLiteral(key), ir)
    return ir


def modify_deep_field(struct, path, new_deep_field, new_struct=None):
    refs = [struct]
    for i in range(len(path)):
        refs[i+1] = Ref(Env.gen_uid(), refs[i].typ[path[i]])

    acc = new_deep_field(refs[-1])
    for parent_struct, field_name in reversed(zip(refs[:-1], path)):
        acc = InsertFields(parent_struct, [(field_name, acc)], None)
    acc = new_struct(acc, refs[-1])
    for struct_ref, field_ref, field_name in reversed(zip(refs[:-1], refs[1:], path)):
        acc = Let(field_ref.name, GetField(struct_ref, field_name))
    return acc


def zip_with_index(array):
    elt = Env.gen_uid()
    inner_row_uid = Env.gen_uid()
    iota = StreamIota(I32(0), I32(1))
    return StreamZip(
        [ToStream(array), iota],
        [elt, inner_row_uid],
        MakeTuple(Ref(elt, array.typ.element_type), Ref(inner_row_uid, tint32)),
        'TakeMinLength')


def zip_with_index_field(array, idx_field_name):
    elt = Env.gen_uid()
    inner_row_uid = Env.gen_uid()
    iota = StreamIota(I32(0), I32(1))
    return StreamZip(
        [ToStream(array), iota],
        [elt, inner_row_uid],
        InsertFields(Ref(elt, array.typ.element_type), [(idx_field_name, Ref(inner_row_uid, tint32))], None),
        'TakeMinLength')


def impute_type_of_partition_interval_array(
        intervals: Optional[List[Any]]
) -> Tuple[Optional[List[Any]], Any]:
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
        hl.Interval(hl.Struct(__point=i.start),
                    hl.Struct(__point=i.end),
                    i.includes_start,
                    i.includes_end)
        for i in intervals
    ]
    struct_intervals_type = hl.tarray(hl.tinterval(hl.tstruct(__point=pt)))
    return struct_intervals, struct_intervals_type


def filter_predicate_with_keep(ir_pred, keep):
    return Coalesce(ir_pred if keep else ApplyUnaryPrimOp('!', ir_pred), FalseIR())


def make_filter_and_replace(filter, find_replace):
    if find_replace is None:
        find = None
        replace = None
    else:
        find, replace = find_replace
    return {
        'filterPattern': filter,
        'findPattern': find,
        'replacePattern': replace
    }


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
