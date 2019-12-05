import builtins
from typing import Callable

import hail as hl
import hail.ir as ir
from hail.expr.expressions import construct_variable, construct_expr, expr_any, to_expr, unify_all, expr_bool
from hail.expr.types import hail_type
from hail.typecheck import anytype, typecheck
from hail.utils.java import Env

# FIXME, infer loop type?
@typecheck(f=anytype, typ=hail_type, exprs=expr_any)
def loop(f: Callable, typ, *exprs):
    """Expression for writing tail recursive expressions.
    Example
    -------
    To find the sum of all the numbers from n=1...10:
    >>> x = hl.experimental.loop(lambda recur, acc, n: hl.cond(n > 10, acc, recur(acc + n, n + 1)), hl.tint32, 0, 0)
    >>> hl.eval(x)
    55

    Notes
    -----
    The first argument to the lambda is a marker for the recursive call.

    Parameters
    ----------
    f : function ( (marker, *args) -> :class:`.Expression`
        Function of one callable marker, denoting where the recursive call (or calls) is located,
        and many `exprs`, the loop variables.
    typ : :obj:`str` or :class:`.HailType`
        Type the loop returns.
    exprs : variable-length args of :class:`.Expression`
        Expressions to initialize the loop values.
    Returns
    -------
    :class:`.Expression`
        Result of the loop with `exprs` as initial loop values.
    """

    loop_name = Env.get_uid()

    def contains_recursive_call(non_recursive):
        if isinstance(non_recursive, ir.Recur) and non_recursive.name == loop_name:
            return True
        return any([contains_recursive_call(c) for c in non_recursive.children])

    def check_tail_recursive(loop_ir):
        print(str(loop_ir))
        if isinstance(loop_ir, ir.If):
            if contains_recursive_call(loop_ir.cond):
                raise TypeError("branch condition can't contain recursive call!")
            check_tail_recursive(loop_ir.cnsq)
            check_tail_recursive(loop_ir.altr)
        elif isinstance(loop_ir, ir.Let):
            if contains_recursive_call(loop_ir.value):
                raise TypeError("bound value used in other expression can't contain recursive call!")
            check_tail_recursive(loop_ir.body)
        elif not isinstance(loop_ir, ir.Recur) and contains_recursive_call(loop_ir):
            raise TypeError("found recursive expression outside of tail position!")

    @typecheck(recur_exprs=expr_any)
    def make_loop(*recur_exprs):
        if len(recur_exprs) != len(exprs):
            raise TypeError('Recursive call in loop has wrong number of arguments')
        err = None
        for i, (rexpr, expr) in enumerate(zip(recur_exprs, exprs)):
            if rexpr.dtype != expr.dtype:
                if err is None:
                    err = 'Type error in recursive call,'
                err += f'\n  at argument index {i}, loop arg type: {expr.dtype}, '
                err += f'recur arg type: {rexpr.dtype}'
        if err is not None:
            raise TypeError(err)
        irs = [expr._ir for expr in recur_exprs]
        indices, aggregations = unify_all(*recur_exprs)
        return construct_expr(ir.Recur(loop_name, irs, typ), typ, indices, aggregations)

    uid_irs = []
    loop_vars = []

    for expr in exprs:
        uid = Env.get_uid()
        loop_vars.append(construct_variable(uid, expr._type, expr._indices, expr._aggregations))
        uid_irs.append((uid, expr._ir))

    loop_f = to_expr(f(make_loop, *loop_vars))
    check_tail_recursive(loop_f._ir)
    indices, aggregations = unify_all(*exprs, loop_f)
    if loop_f.dtype != typ:
        raise TypeError(f"requested type {typ} does not match inferred type {loop_f.typ}")
    return construct_expr(ir.TailLoop(loop_name, loop_f._ir, uid_irs), loop_f.dtype, indices, aggregations)
