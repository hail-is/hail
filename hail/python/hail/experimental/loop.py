import builtins
from typing import Callable

from hail.expr.expressions import construct_variable, construct_expr, expr_any, to_expr, unify_all
from hail.expr.types import hail_type
from hail.ir.ir import Loop, Recur
from hail.typecheck import anytype, typecheck
from hail.utils.java import Env


# FIXME, infer loop type?
@typecheck(f=anytype, typ=hail_type, exprs=expr_any)
def loop(f: Callable, typ, *exprs):
    @typecheck(recur_exprs=expr_any)
    def make_loop(*recur_exprs):
        if len(recur_exprs) != len(exprs):
            raise TypeError('loop and recursion must have the same number of arguments')
        irs = [expr._ir for expr in recur_exprs]
        indices, aggregations = unify_all(*recur_exprs)
        return construct_expr(Recur(irs, typ), typ, indices, aggregations)

    uid_irs = []
    args = []

    for expr in exprs:
        uid = Env.get_uid()
        args.append(construct_variable(uid, expr._type, expr._indices, expr._aggregations))
        uid_irs.append((uid, expr._ir))

    lambda_res = to_expr(f(make_loop, *args))
    indices, aggregations = unify_all(*exprs, lambda_res)
    ir = Loop(uid_irs, lambda_res._ir)
    assert ir.typ == typ, f"requested type {typ} does not match inferred type {ir.typ}"
    return construct_expr(ir, lambda_res.dtype, indices, aggregations)
