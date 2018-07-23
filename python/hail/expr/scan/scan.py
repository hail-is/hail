import hail as hl
from hail.expr.expressions import *
from hail.expr.types import *
from hail.utils import wrap_to_list
from hail.ir import *
from .scan_utils import _scan_func as _agg_func
from ..aggregators.agg_utils import agg_expr, _to_agg


@typecheck(expr=agg_expr(expr_any))
def collect(expr) -> ArrayExpression:
    return _agg_func('Collect', expr, tarray(expr.dtype))


@typecheck(expr=agg_expr(expr_any))
def collect_as_set(expr) -> SetExpression:
    return _agg_func('CollectAsSet', expr, tset(expr.dtype))


@typecheck(expr=nullable(agg_expr(expr_any)))
def count(expr=None) -> Int64Expression:
    if expr is not None:
        return _agg_func('Count', expr, tint64)
    else:
        return _agg_func('Count', _to_agg(hl.int32(0)), tint64)


@typecheck(condition=expr_bool)
def count_where(condition) -> Int64Expression:
    return _agg_func('Count', filter(condition, 0), tint64)


@typecheck(condition=agg_expr(expr_bool))
def any(condition) -> BooleanExpression:
    return count(filter(lambda x: x, condition)) > 0


@typecheck(condition=agg_expr(expr_bool))
def all(condition) -> BooleanExpression:
    n_defined = count(filter(lambda x: hl.is_defined(x), condition))
    n_true = count(filter(lambda x: hl.is_defined(x) & x, condition))
    return n_defined == n_true


@typecheck(expr=agg_expr(expr_any))
def counter(expr) -> DictExpression:
    return _agg_func('Counter', expr, tdict(expr.dtype, tint64))


@typecheck(expr=agg_expr(expr_any),
           n=int,
           ordering=nullable(oneof(expr_any, func_spec(1, expr_any))))
def take(expr, n, ordering=None) -> ArrayExpression:
    n = to_expr(n)
    if ordering is None:
        return _agg_func('Take', expr, tarray(expr.dtype), [n])
    elif callable(ordering):
        return _agg_func('TakeBy', expr, tarray(expr.dtype), [n], f=ordering)
    else:
        return _agg_func('TakeBy', expr, tarray(expr.dtype), [n], f=lambda x: ordering)


@typecheck(expr=agg_expr(expr_numeric))
def min(expr) -> NumericExpression:
    return _agg_func('Min', expr, expr.dtype)


@typecheck(expr=agg_expr(expr_numeric))
def max(expr) -> NumericExpression:
    return _agg_func('Max', expr, expr.dtype)


@typecheck(expr=agg_expr(expr_oneof(expr_int64, expr_float64)))
def sum(expr):
    return _agg_func('Sum', expr, expr.dtype)


@typecheck(expr=agg_expr(expr_array(expr_oneof(expr_int64, expr_float64))))
def array_sum(expr) -> ArrayExpression:
    return _agg_func('Sum', expr, expr.dtype)


@typecheck(expr=agg_expr(expr_float64))
def mean(expr) -> Float64Expression:
    return sum(expr)/count(expr)


@typecheck(expr=agg_expr(expr_float64))
def stats(expr) -> StructExpression:
    return _agg_func('Statistics', expr, tstruct(mean=tfloat64,
                                                 stdev=tfloat64,
                                                 min=tfloat64,
                                                 max=tfloat64,
                                                 n=tint64,
                                                 sum=tfloat64))


@typecheck(expr=agg_expr(expr_oneof(expr_int64, expr_float64)))
def product(expr):
    return _agg_func('Product', expr, expr.dtype)


@typecheck(predicate=agg_expr(expr_bool))
def fraction(predicate) -> Float64Expression:
    return _agg_func("Fraction", predicate, tfloat64)


@typecheck(expr=agg_expr(expr_call))
def hardy_weinberg(expr) -> StructExpression:
    t = tstruct(r_expected_het_freq=tfloat64, p_hwe=tfloat64)
    return _agg_func('HardyWeinberg', expr, t)


@typecheck(expr=agg_expr(expr_oneof(expr_array(), expr_set())))
def explode(expr) -> Aggregable:
    return expr._flatmap(identity)


@typecheck(condition=oneof(func_spec(1, expr_bool), expr_bool), expr=agg_expr(expr_any))
def filter(condition, expr) -> Aggregable:
    f = condition if callable(condition) else lambda x: condition
    return expr._filter(f)


@typecheck(f=oneof(func_spec(1, expr_any), expr_any), expr=agg_expr(expr_any))
def map(f, expr) -> Aggregable:
    f2 = f if callable(f) else lambda x: f
    return expr._map(f2)


@typecheck(f=oneof(func_spec(1, expr_array()), expr_array()), expr=agg_expr(expr_any))
def flatmap(f, expr) -> Aggregable:
    f2 = f if callable(f) else lambda x: f
    return expr._flatmap(f2)


@typecheck(expr=agg_expr(expr_call), prior=expr_float64)
def inbreeding(expr, prior) -> StructExpression:
    t = tstruct(f_stat=tfloat64,
                n_called=tint64,
                expected_homs=tfloat64,
                observed_homs=tint64)
    return _agg_func('Inbreeding', expr, t, f=lambda x: prior)


@typecheck(call=agg_expr(expr_call), alleles=expr_array(expr_str))
def call_stats(call, alleles) -> StructExpression:
    n_alleles = hl.len(alleles)
    t = tstruct(AC=tarray(tint32),
                AF=tarray(tfloat64),
                AN=tint32,
                homozygote_count=tarray(tint32))

    return _agg_func('CallStats', call, t, [], init_op_args=[n_alleles])


@typecheck(expr=agg_expr(expr_float64), start=expr_float64, end=expr_float64, bins=expr_int32)
def hist(expr, start, end, bins) -> StructExpression:
    t = tstruct(bin_edges=tarray(tfloat64),
                bin_freq=tarray(tint64),
                n_smaller=tint64,
                n_larger=tint64)
    return _agg_func('Histogram', expr, t, [start, end, bins])


@typecheck(gp=agg_expr(expr_array(expr_float64)))
def info_score(gp) -> StructExpression:
    t = hl.tstruct(score=hl.tfloat64, n_included=hl.tint32)
    return _agg_func('InfoScore', gp, t)


@typecheck(y=agg_expr(expr_float64),
           x=oneof(expr_float64, sequenceof(expr_float64)))
def linreg(y, x):
    x = wrap_to_list(x)
    k = len(x)
    if k == 0:
        raise ValueError("'linreg' requires at least one predictor in `x`")

    t = hl.tstruct(beta=hl.tarray(hl.tfloat64),
                   standard_error=hl.tarray(hl.tfloat64),
                   t_stat=hl.tarray(hl.tfloat64),
                   p_value=hl.tarray(hl.tfloat64),
                   n=hl.tint64)

    x = hl.array(x)
    k = hl.int32(k)

    return _agg_func('LinearRegression', y, t, [k], f=lambda expr: x)
