from hail.utils.java import Env
from .ir import *


def filter_predicate_with_keep(ir_pred, keep):
    pred = Env.get_uid()
    return Let(pred,
               ir_pred if keep else ApplyUnaryPrimOp('!', ir_pred),
                  If(IsNA(Ref(pred)),
                     FalseIR(),
                     Ref(pred)))

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