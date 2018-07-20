from hail.expr.expressions import *
from hail.ir import *


class AggregableChecker(TypeChecker):
    def __init__(self, coercer):
        self.coercer = coercer
        super(AggregableChecker, self).__init__()

    def expects(self):
        return self.coercer.expects()

    def format(self, arg):
        if isinstance(arg, Aggregable):
            return f'<aggregable Expression of type {repr(arg.dtype)}>'
        else:
            return self.coercer.format(arg)

    def check(self, x, caller, param):
        coercer = self.coercer
        if isinstance(x, Aggregable):
            if coercer.can_coerce(x.dtype):
                if coercer.requires_conversion(x.dtype):
                    return x._map(lambda x_: coercer.coerce(x_))
                else:
                    return x
            else:
                raise TypecheckFailure
        else:
            x = coercer.check(x, caller, param)
            return _to_agg(x)


def _to_agg(x):
    return Aggregable(x._ir, x._type, x._indices, x._aggregations)


agg_expr = AggregableChecker


@typecheck(name=str,
           aggregable=Aggregable,
           ret_type=hail_type,
           constructor_args=sequenceof(expr_any),
           init_op_args=nullable(sequenceof(expr_any)),
           f=nullable(func_spec(1, expr_any)))
def _agg_func(name, aggregable, ret_type, constructor_args=(), init_op_args=None, f=None):
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

    ir = ApplyAggOp(seq_op._ir,
                    list(map(get_ir, constructor_args)),
                    None if init_op_args is None else list(map(get_ir, init_op_args)),
                    signature)
    indices, _ = unify_all(*args)
    return construct_expr(ir, ret_type, indices, aggregations.push(Aggregation(aggregable, *args)))


def _check_agg_bindings(expr):
    bound_references = {ref.name for ref in expr._ir.search(lambda ir: isinstance(ir, Ref) and not isinstance(ir, TopLevelReference))}
    free_variables = bound_references - expr._ir.bound_variables
    if free_variables:
        raise ExpressionException("dynamic variables created by 'hl.bind' or lambda methods like 'hl.map' may not be aggregated")