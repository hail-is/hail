from typing import Callable

from hail import ir
from hail.expr.expressions import construct_expr, construct_variable, expr_any, to_expr, unify_all
from hail.expr.types import hail_type
from hail.typecheck import anytype, typecheck
from hail.utils.java import Env


@typecheck(f=anytype, typ=hail_type, args=expr_any)
def loop(f: Callable, typ, *args):
    r"""Define and call a tail-recursive function with given arguments.

    Notes
    -----
    The argument `f` must be a function where the first argument defines the
    recursive call, and the remaining arguments are the arguments to the
    recursive function, e.g. to define the recursive function

    .. math::

        f(x, y) = \begin{cases}
        y & \textrm{if } x \equiv 0 \\
        f(x - 1, y + x) & \textrm{otherwise}
        \end{cases}


    we would write:
    >>> f = lambda recur, x, y: hl.if_else(x == 0, y, recur(x - 1, y + x))

    Full recursion is not supported, and any non-tail-recursive methods will
    throw an error when called.

    This means that the result of any recursive call within the function must
    also be the result of the entire function, without modification. Let's
    consider two different recursive definitions for the triangle function
    :math:`f(x) = 0 + 1 + \dots + x`:

    >>> def triangle1(x):
    ...     if x == 1:
    ...         return x
    ...     return x + triangle1(x - 1)

    >>> def triangle2(x, total):
    ...     if x == 0:
    ...         return total
    ...     return triangle2(x - 1, total + x)

    The first function definition, `triangle1`, will call itself and then add x.
    This is an example of a non-tail recursive function, since `triangle1(9)`
    needs to modify the result of the inner recursive call to `triangle1(8)` by
    adding 9 to the result.

    The second function is tail recursive: the result of `triangle2(9, 0)` is
    the same as the result of the inner recursive call, `triangle2(8, 9)`.

    Example
    -------
    To find the sum of all the numbers from n=1...10:
    >>> triangle_f = lambda f, x, total: hl.if_else(x == 0, total, f(x - 1, total + x))
    >>> x = hl.experimental.loop(triangle_f, hl.tint32, 10, 0)
    >>> hl.eval(x)
    55

    Let's say we want to find the root of a polynomial equation:
    >>> def polynomial(x):
    ...     return 5 * x**3 - 2 * x - 1

    We'll use `Newton's method<https://en.wikipedia.org/wiki/Newton%27s_method>`
    to find it, so we'll also define the derivative:

    >>> def derivative(x):
    ...     return 15 * x**2 - 2

    and starting at :math:`x_0 = 0`, we'll compute the next step :math:`x_{i+1} = x_i - \frac{f(x_i)}{f'(x_i)}`
    until the difference between :math:`x_{i}` and :math:`x_{i+1}` falls below
    our convergence threshold:

    >>> threshold = 0.005
    >>> def find_root(f, guess, error):
    ...     converged = hl.is_defined(error) & (error < threshold)
    ...     new_guess = guess - (polynomial(guess) / derivative(guess))
    ...     new_error = hl.abs(new_guess - guess)
    ...     return hl.if_else(converged, guess, f(new_guess, new_error))
    >>> x = hl.experimental.loop(find_root, hl.tfloat, 0.0, hl.missing(hl.tfloat))
    >>> hl.eval(x)
    0.8052291984599675

    Warning
    -------
    Using arguments of a type other than numeric types and booleans can cause
    memory issues if if you expect the recursive call to happen many times.

    Parameters
    ----------
    f : function ( (marker, \*args) -> :class:`.Expression`
        Function of one callable marker, denoting where the recursive call (or calls) is located,
        and many `args`, the loop variables.
    typ : :class:`str` or :class:`.HailType`
        Type the loop returns.
    args : variable-length args of :class:`.Expression`
        Expressions to initialize the loop values.
    Returns
    -------
    :class:`.Expression`
        Result of the loop with `args` as initial loop values.
    """

    loop_name = Env.get_uid()

    def contains_recursive_call(non_recursive):
        if isinstance(non_recursive, ir.Recur) and non_recursive.name == loop_name:
            return True
        return any([contains_recursive_call(c) for c in non_recursive.children])

    def check_tail_recursive(loop_ir):
        if isinstance(loop_ir, ir.If):
            if contains_recursive_call(loop_ir.cond):
                raise TypeError("branch condition can't contain recursive call!")
            check_tail_recursive(loop_ir.cnsq)
            check_tail_recursive(loop_ir.altr)
        elif isinstance(loop_ir, ir.Let):
            if contains_recursive_call(loop_ir.value):
                raise TypeError("bound value used in other expression can't contain recursive call!")
            check_tail_recursive(loop_ir.body)
        elif isinstance(loop_ir, ir.TailLoop):
            if any(contains_recursive_call(x) for n, x in loop_ir.params):
                raise TypeError("parameters passed to inner loop can't contain recursive call!")
        elif not isinstance(loop_ir, ir.Recur) and contains_recursive_call(loop_ir):
            raise TypeError("found recursive expression outside of tail position!")

    @typecheck(recur_exprs=expr_any)
    def make_loop(*recur_exprs):
        if len(recur_exprs) != len(args):
            raise TypeError('Recursive call in loop has wrong number of arguments')
        err = None
        for i, (rexpr, expr) in enumerate(zip(recur_exprs, args)):
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

    for expr in args:
        uid = Env.get_uid()
        loop_vars.append(construct_variable(uid, expr._type, expr._indices, expr._aggregations))
        uid_irs.append((uid, expr._ir))

    loop_f = to_expr(f(make_loop, *loop_vars))
    if loop_f.dtype != typ:
        raise TypeError(f"requested type {typ} does not match inferred type {loop_f.dtype}")
    check_tail_recursive(loop_f._ir)
    indices, aggregations = unify_all(*args, loop_f)

    return construct_expr(ir.TailLoop(loop_name, loop_f._ir, uid_irs), loop_f.dtype, indices, aggregations)
