from __future__ import print_function  # Python 2 and 3 print compatibility
from hail.typecheck import *
from hail.expr.expression import *
from hail.expr.ast import *
from hail.genetics import Variant, Locus, Call, GenomeReference

expr_int32 = oneof(Int32Expression, int)
expr_numeric = oneof(Float32Expression, Float64Expression, Int64Expression, float, expr_int32)
expr_list = oneof(list, ArrayExpression)
expr_set = oneof(set, SetExpression)
expr_bool = oneof(bool, BooleanExpression)
expr_struct = oneof(Struct, StructExpression)
expr_str = oneof(strlike, StringExpression)
expr_variant = oneof(Variant, VariantExpression)
expr_locus = oneof(Locus, LocusExpression)
expr_call = oneof(Call, CallExpression)


def _func(name, ret_type, *args):
    indices, aggregations, joins = unify_all(*args)
    return construct_expr(ApplyMethod(name, *(a._ast for a in args)), ret_type, indices, aggregations, joins)

@typecheck(t=Type)
def null(t):
    return Expression(Literal('NA: {}'.format(t)), t)


def capture(x):
    return to_expr(x)


def broadcast(x):
    expr = to_expr(x)
    uid = Env._get_uid()

    def joiner(obj):
        from hail.api2.table import Table
        from hail.api2.matrixtable import MatrixTable
        if isinstance(obj, Table):
            return Table(obj._hc, obj._jkt.annotateGlobalExpr('{} = {}'.format(uid, expr._ast.to_hql())))
        else:
            assert isinstance(obj, MatrixTable)
            return MatrixTable(obj._hc, obj._jvds.annotateGlobalExpr('global.{} = {}'.format(uid, expr._ast.to_hql())))

    return construct_expr(GlobalJoinReference(uid), expr._type, joins=(Join(joiner, [uid]),))


@args_to_expr
@typecheck(c1=expr_int32, c2=expr_int32, c3=expr_int32, c4=expr_int32)
def chisq(c1, c2, c3, c4):
    ret_type = TStruct(['pValue', 'oddsRatio'], [TFloat64(), TFloat64()])
    return _func("chisq", ret_type, c1, c2, c3, c4)


@args_to_expr
@typecheck(left=expr_variant, right=expr_variant)
def combine_variants(left, right):
    if not left._type._rg == right._type._rg:
        raise TypeError('Reference genome mismatch: {}, {}'.format(left._type._rg, right._type._rg))
    ret_type = TStruct(['variant', 'laIndices', 'raIndices'],
                       [left._type, TDict(TInt32(), TInt32()), TDict(TInt32(), TInt32())])
    return _func("combineVariants", ret_type, left, right)


@typecheck(c1=expr_int32, c2=expr_int32, c3=expr_int32, c4=expr_int32, min_cell_count=expr_int32)
@args_to_expr
def ctt(c1, c2, c3, c4, min_cell_count):
    ret_type = TStruct(['pValue', 'oddsRatio'], [TFloat64(), TFloat64()])
    return _func("ctt", ret_type, c1, c2, c3, c4, min_cell_count)


@typecheck(keys=expr_list, values=expr_list)
@args_to_expr
def Dict(keys, values):
    key_col = to_expr(keys)
    value_col = to_expr(values)
    ret_type = TDict(key_col._type, value_col._type)
    return _func("Dict", ret_type, keys, values)


@typecheck(x=expr_numeric, lamb=expr_numeric, logP=expr_bool)
@args_to_expr
def dpois(x, lamb, logP=False):
    return _func("dpois", TFloat64(), x, lamb, logP)


@typecheck(s=oneof(Struct, StructExpression), identifiers=tupleof(expr_str))
def drop(s, *identifiers):
    s = to_expr(s)
    ret_type = s._type._drop(*identifiers)
    return construct_expr(StructOp('drop', s._ast, *identifiers),
                          ret_type, s._indices, s._aggregations, s._joins)


@typecheck(x=expr_numeric)
@args_to_expr
def exp(x):
    return _func("exp", TFloat64(), x)


@typecheck(c1=expr_int32, c2=expr_int32, c3=expr_int32, c4=expr_int32)
@args_to_expr
def fet(c1, c2, c3, c4):
    ret_type = TStruct(['pValue', 'oddsRatio', 'ci95Lower', 'ci95Upper'],
                       [TFloat64(), TFloat64(), TFloat64(), TFloat64()])
    return _func("fet", ret_type, c1, c2, c3, c4)


@typecheck(j=expr_int32, k=expr_int32)
@args_to_expr
def gt_index(j, k):
    return _func("gtIndex", TInt32(), j, k)


@typecheck(j=expr_int32)
@args_to_expr
def gtj(j):
    return _func("gtj", TInt32(), j)


@typecheck(k=expr_int32)
@args_to_expr
def gtk(k):
    return _func("gtk", TInt32(), k)


@typecheck(num_hom_ref=expr_int32, num_het=expr_int32, num_hom_var=expr_int32)
@args_to_expr
def hwe(num_hom_ref, num_het, num_hom_var):
    ret_type = TStruct(['rExpectedHetFrequency', 'pHWE'], [TFloat64(), TFloat64()])
    return _func("hwe", ret_type, num_hom_ref, num_het, num_hom_var)


@typecheck(structs=oneof(ArrayStructExpression, listof(Struct)),
           identifier=strlike)
def index(structs, identifier):
    structs = to_expr(structs)
    struct_type = structs._type.element_type
    struct_fields = {fd.name: fd.typ for fd in struct_type.fields}

    if identifier not in struct_fields:
        raise RuntimeError("`structs' does not have a field with identifier `{}'. " \
                           "Struct type is {}.".format(identifier, struct_type))

    key_type = struct_fields[identifier]
    value_type = struct_type._drop(identifier)

    ast = StructOp('index', structs._ast, identifier)
    return construct_expr(ast, TDict(key_type, value_type),
                          structs._indices, structs._aggregations, structs._joins)


@typecheck(contig=expr_str, pos=expr_int32, reference_genome=nullable(GenomeReference))
def locus(contig, pos, reference_genome=None):
    contig = to_expr(contig)
    pos = to_expr(pos)
    if reference_genome is None:
        reference_genome = Env.hc().default_reference
    indices, aggregations, joins = unify_all(contig, pos)
    return construct_expr(ApplyMethod('Locus({})'.format(reference_genome.name), contig._ast, pos._ast),
                          TLocus(reference_genome), indices, aggregations, joins)


@typecheck(s=expr_str, reference_genome=nullable(GenomeReference))
def parse_locus(s, reference_genome=None):
    s = to_expr(s)
    if reference_genome is None:
        reference_genome = Env.hc().default_reference
    return construct_expr(ApplyMethod('Locus({})'.format(reference_genome.name), s._ast), TLocus(reference_genome),
                          s._indices, s._aggregations, s._joins)


@typecheck(start=expr_locus, end=expr_locus)
def interval(start, end):
    start = to_expr(start)
    end = to_expr(end)

    indices, aggregations, joins = unify_all(start, end)
    if not start._type._rg == end._type._rg:
        raise TypeError('Reference genome mismatch: {}, {}'.format(start._type._rg, end._type._rg))
    return construct_expr(
        ApplyMethod('Interval({})'.format(start._type._rg.name), start._ast, end._ast), TInterval(start._type._rg),
        indices, aggregations, joins)


@typecheck(s=expr_str, reference_genome=nullable(GenomeReference))
def parse_interval(s, reference_genome=None):
    s = to_expr(s)
    if reference_genome is None:
        reference_genome = Env.hc().default_reference
    return construct_expr(
        ApplyMethod('Interval({})'.format(reference_genome.name), s._ast), TInterval(reference_genome),
        s._indices, s._aggregations, s._joins)


@typecheck(contig=expr_str, pos=expr_int32, ref=expr_str, alts=oneof(expr_str, listof(expr_str), expr_list),
           reference_genome=nullable(GenomeReference))
def variant(contig, pos, ref, alts, reference_genome=None):
    contig = to_expr(contig)
    pos = to_expr(pos)
    ref = to_expr(ref)
    alts = to_expr(alts)
    if reference_genome is None:
        reference_genome = Env.hc().default_reference
    indices, aggregations, joins = unify_all(contig, pos, ref, alts)
    return VariantExpression(
        ApplyMethod('Variant({})'.format(reference_genome.name),
                    contig._ast, pos._ast, ref._ast, alts._ast),
        TVariant(reference_genome), indices, aggregations, joins)


@typecheck(s=expr_str, reference_genome=nullable(GenomeReference))
def parse_variant(s, reference_genome=None):
    s = to_expr(s)
    if reference_genome is None:
        reference_genome = Env.hc().default_reference
    return construct_expr(ApplyMethod('Variant({})'.format(reference_genome.name), s._ast),
                          TVariant(reference_genome), s._indices, s._aggregations, s._joins)


@args_to_expr
def call(i):
    return CallExpression(ApplyMethod('Call', i._ast), TCall(), i._indices, i._aggregations, i._joins)


@typecheck(x=anytype)
@args_to_expr
def is_defined(x):
    return _func("isDefined", TBoolean(), x)


@typecheck(x=anytype)
@args_to_expr
def is_missing(x):
    return _func("isMissing", TBoolean(), x)


@typecheck(x=expr_numeric)
@args_to_expr
def is_nan(x):
    return _func("isnan", TBoolean(), x)


@typecheck(x=anytype)
@args_to_expr
def json(x):
    return _func("json", TString(), x)


@typecheck(x=expr_numeric, b=expr_numeric)
def log(x, b=None):
    x = to_expr(x)
    if b is not None:
        return _func("log", TFloat64(), x, to_expr(b))
    else:
        return _func("log", TFloat64(), x)


@typecheck(x=expr_numeric)
@args_to_expr
def log10(x):
    return _func("log10", TFloat64(), x)


@typecheck(x=expr_bool)
@args_to_expr
def logical_not(x):
    return _func("!", TBoolean(), x)


@typecheck(s1=StructExpression, s2=StructExpression)
@args_to_expr
def merge(s1, s2):
    ret_type = s1._type._merge(s2._type)
    return _func("merge", ret_type, s1, s2)


@typecheck(a=anytype, b=anytype)
@args_to_expr
def or_else(a, b):
    a = to_expr(a)
    # FIXME: type promotion
    return _func("orElse", a._type, a, b)


@typecheck(a=anytype, b=anytype)
@args_to_expr
def or_missing(a, b):
    a = to_expr(a)
    return _func("orMissing", a._type, a, b)


@typecheck(x=expr_numeric, df=expr_numeric)
@args_to_expr
def pchisqtail(x, df):
    return _func("pchisqtail", TFloat64(), x, df)


@typecheck(p=expr_numeric)
@args_to_expr
def pcoin(p):
    return _func("pcoin", TBoolean(), p)


@typecheck(x=expr_numeric)
@args_to_expr
def pnorm(x):
    return _func("pnorm", TFloat64(), x)


@typecheck(x=expr_numeric, lamb=expr_numeric, lower_tail=expr_bool, logP=expr_bool)
@args_to_expr
def ppois(x, lamb, lower_tail=True, logP=False):
    return _func("ppois", TFloat64(), x, lamb, lower_tail, logP)


@typecheck(p=expr_numeric, df=expr_numeric)
@args_to_expr
def qchisqtail(p, df):
    return _func("qchisqtail", TFloat64(), p, df)


@typecheck(p=expr_numeric)
@args_to_expr
def qnorm(p):
    return _func("qnorm", TFloat64(), p)


@typecheck(p=expr_numeric, lamb=expr_numeric, lower_tail=expr_bool, logP=expr_bool)
@args_to_expr
def qpois(p, lamb, lower_tail=True, logP=False):
    return _func("qpois", TInt32(), p, lamb, lower_tail, logP)


@typecheck(stop=expr_int32, start=expr_int32, step=expr_int32)
@args_to_expr
def range(stop, start=0, step=1):
    return _func("range", TArray(TInt32()), start, stop, step)


@typecheck(mean=expr_numeric, sd=expr_numeric)
@args_to_expr
def rnorm(mean, sd):
    return _func("rnorm", TFloat64(), mean, sd)


@typecheck(lamb=expr_numeric, n=integral)
def rpois(lamb, n=1):
    if n > 1:
        return _func("rpois", TArray(TFloat64()), n, lamb)
    else:
        return _func("rpois", TFloat64(), lamb)


@typecheck(min=expr_numeric, max=expr_numeric)
@args_to_expr
def runif(min, max):
    return _func("runif", TFloat64(), min, max)


@typecheck(s=oneof(Struct, StructExpression), identifiers=tupleof(expr_str))
def select(s, *identifiers):
    s = to_expr(s)
    ret_type = s._type._select(*identifiers)
    return construct_expr(StructOp('select', s._ast, *identifiers), ret_type, s._indices, s._aggregations, s._joins)


@typecheck(x=expr_numeric)
@args_to_expr
def sqrt(x):
    return _func("sqrt", TFloat64(), x)


@typecheck(x=anytype)
@args_to_expr
def to_str(x):
    return _func("str", TString(), x)


@typecheck(pred=expr_bool, then_case=anytype, else_case=anytype)
@args_to_expr
def cond(pred, then_case, else_case):
    pred = to_expr(pred)
    then_case = to_expr(then_case)
    else_case = to_expr(else_case)

    indices, aggregations, joins = unify_all(pred, then_case, else_case)
    # TODO: promote types
    return construct_expr(Condition(pred._ast, then_case._ast, else_case._ast), then_case._type, indices, aggregations,
                          joins)


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
