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


