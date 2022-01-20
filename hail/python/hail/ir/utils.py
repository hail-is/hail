from .ir import Coalesce, ApplyUnaryPrimOp, FalseIR

import hail as hl


def check_scale_continuity(scale, dtype, aes_key):
    if scale.is_discrete() and is_continuous_type(dtype):
        raise ValueError(f"Aesthetic {aes_key} has continuous dtype but non continuous scale")
    if not scale.is_discrete() and not is_continuous_type(dtype):
        raise ValueError(f"Aesthetic {aes_key} has non continuous dtype but continuous scale")


def is_continuous_type(dtype):
    return dtype in [hl.tint32, hl.tint64, hl.float32, hl.float64]


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
