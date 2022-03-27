from typing import Optional, List, Any, Tuple
from .ir import Coalesce, ApplyUnaryPrimOp, FalseIR
import hail as hl


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
