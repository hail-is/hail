from functools import wraps, update_wrapper

from hail.expr.expressions import *
from hail.ir import *

@typecheck(name=str,
           aggregable=Aggregable,
           ret_type=hail_type,
           constructor_args=sequenceof(expr_any),
           init_op_args=nullable(sequenceof(expr_any)),
           f=nullable(func_spec(1, expr_any)))
def _scan_func(name, aggregable, ret_type, constructor_args=(), init_op_args=None, f=None):
    args = constructor_args if init_op_args is None else constructor_args + init_op_args
    indices, aggregations = unify_all(aggregable, *args)
    if aggregations:
        raise ExpressionException('Cannot aggregate an already-aggregated expression')
    for a in args:
        _check_agg_bindings(a)
    _check_agg_bindings(aggregable)

    def get_type(expr):
        return expr.dtype

    def get_ir(expr):
        return expr._ir

    signature = None
    def agg_sig(*seq_op_args):
        return AggSignature(name,
                            list(map(get_type, constructor_args)),
                            None if init_op_args is None else list(map(get_type, init_op_args)),
                            list(map(get_type, seq_op_args)))

    if name == "Count":
        def make_seq_op(agg):
            return construct_expr(SeqOp(I32(0), [], agg_sig()), None)
        seq_op = aggregable._transformations(aggregable, make_seq_op)
        signature = agg_sig()
    elif f is None:
        def make_seq_op(agg):
            return construct_expr(SeqOp(I32(0), [get_ir(agg)], agg_sig(aggregable)), None)
        seq_op = aggregable._transformations(aggregable, make_seq_op)
        signature = agg_sig(aggregable)
    else:
        def make_seq_op(agg):
            uid = Env.get_uid()
            ref = construct_variable(uid, get_type(agg), agg._indices)
            result = f(ref)
            body = Let(uid, get_ir(agg), get_ir(result))
            return construct_expr(SeqOp(I32(0), [get_ir(agg), body], agg_sig(agg, result)), None)
        seq_op = aggregable._transformations(aggregable, make_seq_op)
        signature = agg_sig(aggregable, f(aggregable))

    ir = ApplyScanOp(seq_op._ir,
                    list(map(get_ir, constructor_args)),
                    None if init_op_args is None else list(map(get_ir, init_op_args)),
                    signature)
    indices, _ = unify_all(aggregable, *args)
    return construct_expr(ir, ret_type, indices, aggregations)

def _check_agg_bindings(expr):
    bound_references = {ref.name for ref in expr._ir.search(lambda ir: isinstance(ir, Ref) and not isinstance(ir, TopLevelReference))}
    free_variables = bound_references - expr._ir.bound_variables
    if free_variables:
        raise ExpressionException("dynamic variables created by 'hl.bind' or lambda methods like 'hl.map' may not be aggregated")


def _lift_aggs_to_scans(agg_expr):
    import hail.ir as ir
    from hail.expr import construct_expr, Aggregable
    if isinstance(agg_expr, Aggregable):
        return agg_expr

    def agg_to_scan(n):
        if isinstance(n, ir.ApplyAggOp):
            a = n.children[0]
            const_args = n.children[1:len(n.constructor_args) + 1]
            init_args = None if n.init_op_args is None else n.children[len(n.constructor_args) + 1:]
            return ir.ApplyScanOp(a, const_args, init_args, n.agg_sig)
        else:
            return n
    return construct_expr(ir.value_recur(agg_expr._ir, agg_to_scan), agg_expr._type, agg_expr._indices, agg_expr._aggregations)


def scan_decorator(f):
    @wraps(f)
    def wrapper(*args, **kwargs):

        intermediate = f(*args, **kwargs)
        return _lift_aggs_to_scans(intermediate)
    update_wrapper(wrapper, f)
    return wrapper
