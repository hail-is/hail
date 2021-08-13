from .ir import Coalesce, ApplyUnaryPrimOp, FalseIR
import hail as hl

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


def parse_type(string_expr, dtype):
    if dtype == hl.tstr:
        return string_expr
    elif dtype == hl.tint32:
        return hl.int32(string_expr)
    elif dtype == hl.tint64:
        return hl.int64(string_expr)
    elif dtype == hl.tfloat32:
        return hl.hl.tfloat32(string_expr)
    # same for int64, float32, float64, bool
    elif isinstance(dtype, hl.tlocus):
        return hl.parse_locus(string_expr, dtype.reference_genome)
    elif isinstance(dtype, hl.tinterval) and isinstance(dtype.point_type, hl.tlocus):
        return hl.parse_locus_interval(string_expr, dtype.point_type.reference_genome)
    elif isinstance(dtype, hl.tcall):
        return hl.parse_call(string_expr)
    else:
        return hl.parse_json(string_expr, dtype)


