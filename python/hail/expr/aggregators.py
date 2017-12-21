from __future__ import print_function  # Python 2 and 3 print compatibility
from hail.typecheck import *
from hail.expr.expression import *
from hail.expr.ast import *
from hail.genetics import Variant, Locus, Call, GenomeReference


def _to_agg(x):
    if isinstance(x, Aggregable):
        return x
    else:
        x = to_expr(x)
        uid = Env._get_uid()
        ast = LambdaClassMethod('map', uid, AggregableReference(), x._ast)
        return Aggregable(ast, x._type, x._indices, x._aggregations, x._joins)


@typecheck(name=strlike, aggregable=Aggregable, ret_type=Type, args=tupleof(anytype))
def _agg_func(name, aggregable, ret_type, *args):
    args = [to_expr(a) for a in args]
    indices, aggregations, joins = unify_all(aggregable, *args)
    if aggregations:
        raise ValueError('cannot aggregate an already-aggregated expression')

    ast = ClassMethod(name, aggregable._ast, *[a._ast for a in args])
    return construct_expr(ast, ret_type, Indices(source=indices.source), (Aggregation(indices),), joins)


def collect(expr):
    agg = _to_agg(expr)
    return _agg_func('collect', agg, TArray(agg._type))


def collect_as_set(expr):
    agg = _to_agg(expr)
    return _agg_func('collectAsSet', agg, TArray(agg._type))


def count(expr):
    return _agg_func('count', _to_agg(expr), TInt64())


def count_where(condition):
    return count(filter(1, condition))


def counter(expr):
    agg = _to_agg(expr)
    return _agg_func('counter', agg, TDict(agg._type, TInt64()))


def take(expr, n, ordering=None):
    agg = _to_agg(expr)
    n = to_expr(n)
    if ordering is None:
        return _agg_func('take', agg, TArray(agg._type), n)
    else:
        uid = Env._get_uid()
        if callable(ordering):
            lambda_result = to_expr(
                ordering(construct_expr(Reference(uid), agg._type, agg._indices,
                                        agg._aggregations, agg._joins)))
        else:
            lambda_result = to_expr(ordering)
        indices, aggregations, joins = unify_all(agg, lambda_result)
        ast = LambdaClassMethod('takeBy', uid, agg._ast, lambda_result._ast, n._ast)

        if aggregations:
            raise ValueError('cannot aggregate an already-aggregated expression')

        return construct_expr(ast, TArray(agg._type), Indices(source=indices.source), (Aggregation(indices),), joins)


def min(expr):
    agg = _to_agg(expr)
    return _agg_func('min', agg, agg._type)


def max(expr):
    agg = _to_agg(expr)
    return _agg_func('max', agg, agg._type)


def sum(expr):
    agg = _to_agg(expr)
    # FIXME I think this type is wrong
    return _agg_func('sum', agg, agg._type)


def mean(expr):
    return stats(expr).mean


def stats(expr):
    agg = _to_agg(expr)
    return _agg_func('stats', agg, TStruct(['mean', 'stdev', 'min', 'max', 'nNotMissing', 'sum'],
                                           [TFloat64(), TFloat64(), TFloat64(), TFloat64(), TInt64(), TFloat64()]))


def product(expr):
    agg = _to_agg(expr)
    # FIXME I think this type is wrong
    return _agg_func('product', agg, agg._type)


def fraction(expr):
    agg = _to_agg(expr)
    if not isinstance(agg._type, TBoolean):
        raise TypeError(
            "'fraction' aggregator expects an expression of type 'TBoolean', found '{}'".format(agg._type.__class__))

    if agg._aggregations:
        raise ValueError('cannot aggregate an already-aggregated expression')

    uid = Env._get_uid()
    ast = LambdaClassMethod('fraction', uid, agg._ast, Reference(uid))
    return construct_expr(ast, TBoolean(), Indices(source=agg._indices.source), (Aggregation(agg._indices),),
                          agg._joins)


def hardy_weinberg(expr):
    t = TStruct(['rExpectedHetFrequency', 'pHWE'], [TFloat64(), TFloat64()])
    agg = _to_agg(expr)
    if not isinstance(agg._type, TCall):
        raise TypeError("aggregator 'hardy_weinberg' requires an expression of type 'TCall', found '{}'".format(
            agg._type.__class__))
    return _agg_func('hardyWeinberg', agg, t)


@typecheck(expr=oneof(expr_list, expr_set))
def explode(expr):
    agg = _to_agg(expr)
    uid = Env._get_uid()
    return Aggregable(LambdaClassMethod('flatMap', uid, agg._ast, Reference(uid)),
                      agg._type, agg._indices, agg._aggregations, agg._joins)


def filter(expr, condition):
    agg = _to_agg(expr)
    uid = Env._get_uid()

    if callable(condition):
        lambda_result = to_expr(
            condition(
                construct_expr(Reference(uid), agg._type, agg._indices, agg._aggregations, agg._joins)))
    else:
        lambda_result = to_expr(condition)

    if not isinstance(lambda_result._type, TBoolean):
        raise TypeError(
            "'filter' expects the 'condition' argument to be or produce an expression of type 'TBoolean', found '{}'".format(
                lambda_result._type.__class__))
    indices, aggregations, joins = unify_all(agg, lambda_result)
    ast = LambdaClassMethod('filter', uid, agg._ast, lambda_result._ast)
    return Aggregable(ast, agg._type, indices, aggregations, joins)


@typecheck(expr=oneof(Aggregable, expr_call), prior=expr_numeric)
def inbreeding(expr, prior):
    agg = _to_agg(expr)
    prior = to_expr(prior)

    if not isinstance(agg._type, TCall):
        raise TypeError("aggregator 'inbreeding' requires an expression of type 'TCall', found '{}'".format(
            agg._type.__class__))

    uid = Env._get_uid()
    ast = LambdaClassMethod('inbreeding', uid, agg._ast, prior._ast)

    indices, aggregations, joins = unify_all(agg, prior)
    if aggregations:
        raise ValueError('cannot aggregate an already-aggregated expression')

    t = TStruct(['Fstat', 'nTotal', 'nCalled', 'expectedHoms', 'observedHoms'],
                [TFloat64(), TInt64(), TInt64(), TFloat64(), TInt64()])
    return construct_expr(ast, t, Indices(source=indices.source), (Aggregation(indices),), joins)


@typecheck(expr=oneof(Aggregable, expr_call), variant=expr_variant)
def call_stats(expr, variant):
    agg = _to_agg(expr)
    variant = to_expr(variant)

    uid = Env._get_uid()

    if not isinstance(agg._type, TCall):
        raise TypeError("aggregator 'call_stats' requires an expression of type 'TCall', found '{}'".format(
            agg._type.__class__))

    ast = LambdaClassMethod('callStats', uid, agg._ast, variant._ast)
    indices, aggregations, joins = unify_all(agg, variant)

    if aggregations:
        raise ValueError('cannot aggregate an already-aggregated expression')

    t = TStruct(['AC', 'AF', 'AN', 'GC'], [TArray(TInt32()), TArray(TFloat64()), TInt32(), TArray(TInt32())])
    return construct_expr(ast, t, Indices(source=indices.source), (Aggregation(indices),), joins)


def hist(expr, start, end, bins):
    agg = _to_agg(expr)
    # FIXME check types
    t = TStruct(['binEdges', 'binFrequencies', 'nLess', 'nGreater'],
                [TArray(TFloat64()), TArray(TInt64()), TInt64(), TInt64()])
    return _agg_func('hist', agg, t, start, end, bins)
