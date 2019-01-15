from hail.utils.java import Env
from .ir import *


def filter_predicate_with_keep(ir_pred, keep):
    pred = Env.get_uid()
    return Let(pred,
               ir_pred if keep else ApplyUnaryOp('!', ir_pred),
                  If(IsNA(Ref(pred)),
                     FalseIR(),
                     Ref(pred)))
