from .ir import Coalesce, ApplyUnaryPrimOp, FalseIR


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
