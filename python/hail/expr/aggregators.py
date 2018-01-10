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


@typecheck(name=strlike, aggregable=Aggregable, ret_type=Type, args=anytype)
def _agg_func(name, aggregable, ret_type, *args):
    args = [to_expr(a) for a in args]
    indices, aggregations, joins = unify_all(aggregable, *args)
    if aggregations:
        raise ValueError('cannot aggregate an already-aggregated expression')

    ast = ClassMethod(name, aggregable._ast, *[a._ast for a in args])
    return construct_expr(ast, ret_type, Indices(source=indices.source), aggregations.push(Aggregation(indices)), joins)


def collect(expr):
    """Collect records into an array.

    Examples
    --------
    Collect the `ID` field where `HT` is greater than 68:

    .. doctest::

        >>> table1.aggregate(ht_over_68 = agg.collect(agg.filter(table1.HT > 68, table1.ID)))
        Struct(ht_over_68=[2, 3])

    Notes
    -----
    The element order of the resulting array is not guaranteed, and in some
    cases is non-deterministic.

    Use :meth:`collect_as_set` to collect unique items.

    Warning
    -------
    Collecting a large number of items can cause out-of-memory exceptions.

    Parameters
    ----------
    expr : :class:`hail.expr.expression.Expression`
        Expression to collect.

    Returns
    -------
    :class:`hail.expr.expression.ArrayExpression`
        Array of all `expr` records.
    """
    agg = _to_agg(expr)
    return _agg_func('collect', agg, TArray(agg._type))


def collect_as_set(expr):
    """Collect records into a set.

    Examples
    --------
    Collect the unique `ID` field where `HT` is greater than 68:

    .. doctest::

        >>> table1.aggregate(ht_over_68 = agg.collect_as_set(agg.filter(table1.HT > 68, table1.ID)))
        Struct(ht_over_68=set([2, 3])

    Warning
    -------
    Collecting a large number of items can cause out-of-memory exceptions.

    Parameters
    ----------
    expr : :class:`hail.expr.expression.Expression`
        Expression to collect.

    Returns
    -------
    :class:`hail.expr.expression.SetExpression`
        Set of unique `expr` records.
    """

    agg = _to_agg(expr)
    return _agg_func('collectAsSet', agg, TArray(agg._type))


def count(expr=None):
    """Count the number of records.

    Examples
    --------
    Group by the `SEX` field and count the number of rows in each category:

    .. doctest::

        >>> (table1.group_by(table1.SEX)
        ...        .aggregate(n=agg.count())
        ...        .show())
        +--------+-------+
        | SEX    |     n |
        +--------+-------+
        | String | Int64 |
        +--------+-------+
        | M      |     2 |
        | F      |     2 |
        +--------+-------+

    Notes
    -----
    If `expr` is not provided, then this method will count the number of
    records aggregated. If `expr` is provided, then the result should
    make use of :meth:`filter` or :meth:`explode` so that the number of
    records aggregated changes.

    Parameters
    ----------
    expr : :class:`.Expression`, or :obj:`None`
        Expression to count.

    Returns
    -------
    :class:`hail.expr.expression.Int64Expression`
        Total number of records.
    """
    if expr is not None:
        return _agg_func('count', _to_agg(expr), TInt64())
    else:
        return _agg_func('count', _to_agg(0), TInt64())


def count_where(condition):
    """Count the number of records where a predicate is ``True``.

    Examples
    --------

    Count the number of individuals with `HT` greater than 68:

    .. doctest::

        >>> table1.aggregate(ht_over_68 = agg.count_where(table1.HT > 68))
        Struct(ht_over_68=2)

    Parameters
    ----------
    condition : :class:`hail.expr.expression.BooleanExpression`
        Criteria for inclusion.

    Returns
    -------
    :class:`hail.expr.expression.Int64Expression`
        Total number of records where `condition` is ``True``.
    """
    return _agg_func('count', filter(condition, 0), TInt64())


def counter(expr):
    """Count the occurrences of each unique record and return a dictionary.

    Examples
    --------
    Count the number of individuals for each unique `SEX` value:

    .. doctest::

        >>> table1.aggregate(sex_counts = agg.counter(table1.SEX))
        Struct(sex_counts={u'M': 2L, u'F': 2L})

    Notes
    -----
    This aggregator method returns a dict expression whose key type is the
    same type as `expr` and whose value type is :class:`Int64Expression`.
    This dict contains a key for each unique value of `expr`, and the value
    is the number of times that key was observed.

    Ensure that the result can be stored in memory on a single machine.

    Warning
    -------
    Using :meth:`counter` with a large number of unique items can cause
    out-of-memory exceptions.

    Parameters
    ----------
    expr : :class:`hail.expr.expression.Expression`
        Expression to count by key.

    Returns
    -------
    :class:`hail.expr.expression.DictExpression`
        Dictionary with the number of occurrences of each unique record.
    """
    agg = _to_agg(expr)
    return _agg_func('counter', agg, TDict(agg._type, TInt64()))


def take(expr, n, ordering=None):
    """Take `n` records of `expr`, optionally ordered by `ordering`.

    Examples
    --------
    Take 3 elements of field `X`:

    .. doctest::

        >>> table1.aggregate(xs = agg.take(table1.X, 3))
        Struct(xs=[5, 6, 7])

    Take the `ID` and `HT` fields, ordered by `HT` (descending):

    .. doctest::

        >>> table1.aggregate(hts = agg.take(Struct(ID=table1.ID, HT=table1.HT),
        ...                                 3,
        ...                                 ordering=-table1.HT))
        Struct(hts=[Struct(ID=2, HT=72), Struct(ID=3, HT=70), Struct(ID=1, HT=65)])

    Notes
    -----
    The resulting array can include fewer than `n` elements if there are fewer
    than `n` total records.

    The `ordering` argument may be an expression, a function, or ``None``.

    If `ordering` is an expression, this expression's type should be one with
    a natural ordering (e.g. numeric).

    If `ordering` is a function, it will be evaluated on each record of `expr`
    to compute the value used for ordering. In the above example,
    ``ordering=-table1.HT`` and ``ordering=lambda x: -x.HT`` would be
    equivalent.

    If `ordering` is ``None``, then there is no guaranteed ordering on the
    elements taken, and and the results may be non-deterministic.

    Missing values are always sorted **last**.

    Parameters
    ----------
    expr : :class:`hail.expr.expression.Expression`
        Expression to store.
    n : :class:`hail.expr.expression.Int32Expression`
        Number of records to take.
    ordering : :class:`hail.expr.expression.Expression` or function or None
        Optional ordering on records.

    Returns
    -------
    :class:`hail.expr.expression.ArrayExpression`
        Array of up to `n` records of `expr`.

    """
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

        return construct_expr(ast, TArray(agg._type), Indices(source=indices.source),
                              aggregations.push(Aggregation(indices)), joins)


def min(expr):
    """Compute the minimum `expr`.

    Examples
    --------
    Compute the minimum value of `HT`:

    .. doctest::

        >>> table1.aggregate(min_ht = agg.min(table1.HT))
        Struct(min_ht=60)

    Notes
    -----
    This method returns the minimum non-missing value. If there are no
    non-missing values, then the result is missing.

    Parameters
    ----------
    expr : :class:`hail.expr.expression.NumericExpression`
        Numeric expression.

    Returns
    -------
    :class:`hail.expr.expression.NumericExpression`
        Minimum value of all `expr` records, same type as `expr`.
    """
    agg = _to_agg(expr)
    return _agg_func('min', agg, agg._type)


def max(expr):
    """Compute the maximum `expr`.

    Examples
    --------
    Compute the maximum value of `HT`:

    .. doctest::

        >>> table1.aggregate(max_ht = agg.max(table1.HT))
        Struct(max_ht=72)

    Notes
    -----
    This method returns the maximum non-missing value. If there are no
    non-missing values, then the result is missing.

    Parameters
    ----------
    expr : :class:`hail.expr.expression.NumericExpression`
        Numeric expression.

    Returns
    -------
    :class:`hail.expr.expression.NumericExpression`
        Maximum value of all `expr` records, same type as `expr`.
    """
    agg = _to_agg(expr)
    return _agg_func('max', agg, agg._type)


def sum(expr):
    """Compute the sum of all records of `expr`.

    Examples
    --------
    Compute the sum of field `C1`:

    .. doctest::

        >>> table1.aggregate(c1_sum = agg.sum(table1.C1))
        Struct(c1_sum=25)

    Notes
    -----
    Missing values are ignored (treated as zero).

    If `expr` is an expression of type :class:`hail.expr.types.TInt32` or :class:`hail.expr.types.TInt64`, then
    the result is an expression of type :class:`hail.expr.types.TInt64`. If `expr` is an
    expression of type :class:`hail.expr.types.TFloat32` or :class:`hail.expr.types.TFloat64`, then the result
    is an expression of type :class:`hail.expr.types.TFloat64`.

    Parameters
    ----------
    expr : :class:`hail.expr.expression.NumericExpression`
        Numeric expression.

    Returns
    -------
    :class:`hail.expr.expression.Int64Expression` or :class:`hail.expr.expression.Float64Expression`
        Sum of records of `expr`.
    """
    agg = _to_agg(expr)
    # FIXME I think this type is wrong
    return _agg_func('sum', agg, agg._type)


def mean(expr):
    """Compute the mean value of records of `expr`.

    Examples
    --------
    Compute the mean of field `HT`:

    .. doctest::

        >>> table1.aggregate(mean_ht = agg.mean(table1.HT))
        Struct(mean_ht=66.75)

    Notes
    -----
    Missing values are ignored.

    Parameters
    ----------
    expr : :class:`hail.expr.expression.NumericExpression`
        Numeric expression.

    Returns
    -------
    :class:`hail.expr.expression.Float64Expression`
        Mean value of records of `expr`.
    """
    return stats(expr).mean


def stats(expr):
    """Compute a number of useful statistics about `expr`.

    Examples
    --------
    Compute statistics about field `HT`:

    .. doctest::

        >>> table1.aggregate(ht_stats = agg.stats(table1.HT))
        Struct(ht_stats=Struct(min=60.0, max=72.0, sum=267.0, stdev=4.65698400255, nNotMissing=4, mean=66.75))

    Notes
    -----
    Computes a struct with the following fields:

    - `min` (:class:`hail.expr.types.TFloat64`) - Minimum value.
    - `max` (:class:`hail.expr.types.TFloat64`) - Maximum value.
    - `mean` (:class:`hail.expr.types.TFloat64`) - Mean value,
    - `stdev` (:class:`hail.expr.types.TFloat64`) - Standard deviation.
    - `nNotMissing` (:class:`hail.expr.types.TFloat64`) - Number of non-missing records.
    - `sum` (:class:`hail.expr.types.TFloat64`) - Sum.

    Parameters
    ----------
    expr : :class:`hail.expr.expression.NumericExpression`
        Numeric expression.

    Returns
    -------
    :class:`hail.expr.expression.StructExpression`
        Struct expression with fields `mean`, `stdev`, `min`, `max`,
        `nNotMissing`, and `sum`.
    """
    agg = _to_agg(expr)
    return _agg_func('stats', agg, TStruct(['mean', 'stdev', 'min', 'max', 'nNotMissing', 'sum'],
                                           [TFloat64(), TFloat64(), TFloat64(), TFloat64(), TInt64(), TFloat64()]))


def product(expr):
    """Compute the product of all records of `expr`.

    Examples
    --------
    Compute the product of field `C1`:

    .. doctest::

        >>> table1.aggregate(c1_prod = agg.product(table1.C1))
        Struct(c1_prod=440)

    Notes
    -----
    Missing values are ignored (treated as one).

    If `expr` is an expression of type :class:`hail.expr.types.TInt32` or :class:`hail.expr.types.TInt64`, then
    the result is an expression of type :class:`hail.expr.types.TInt64`. If `expr` is an
    expression of type :class:`hail.expr.types.TFloat32` or :class:`hail.expr.types.TFloat64`, then the result
    is an expression of type :class:`hail.expr.types.TFloat64`.

    Parameters
    ----------
    expr : :class:`hail.expr.expression.NumericExpression`
        Numeric expression.

    Returns
    -------
    :class:`hail.expr.expression.Int64Expression` or :class:`hail.expr.expression.Float64Expression`
        Product of records of `expr`.
    """

    agg = _to_agg(expr)
    # FIXME I think this type is wrong
    return _agg_func('product', agg, agg._type)


def fraction(predicate):
    """Compute the fraction of records where `predicate` is ``True``.

    Examples
    --------
    Compute the fraction of rows where `SEX` is "F" and `HT` > 65:

    .. doctest::

        >>> table1.aggregate(frac = agg.fraction((table1.SEX == 'F') & (table1.HT > 65)))
        Struct(frac=0.25)

    Notes
    -----
    Missing values for `predicate` are treated as ``False``.

    Parameters
    ----------
    predicate : :class:`hail.expr.expression.BooleanExpression`
        Boolean predicate.

    Returns
    -------
    :class:`hail.expr.expression.Float64Expression`
        Fraction of records where `predicate` is ``True``.
    """
    agg = _to_agg(predicate)
    if not isinstance(agg._type, TBoolean):
        raise TypeError(
            "'fraction' aggregator expects an expression of type 'TBoolean', found '{}'".format(agg._type.__class__))

    if agg._aggregations:
        raise ValueError('cannot aggregate an already-aggregated expression')

    uid = Env._get_uid()
    ast = LambdaClassMethod('fraction', uid, agg._ast, Reference(uid))
    return construct_expr(ast, TFloat64(), Indices(source=agg._indices.source),
                          agg._aggregations.push(Aggregation(agg._indices)), agg._joins)


def hardy_weinberg(expr):
    """Compute Hardy-Weinberg Equilbrium (HWE) p-value and heterozygosity ratio.

    Examples
    --------
    Compute HWE statistics per row of a dataset:

    .. doctest::

        >>> dataset_result = dataset.annotate_rows(hwe = agg.hardy_weinberg(dataset.GT))

    Compute HWE statistics for a single population:

    .. doctest::

        >>> dataset_result = dataset.annotate_rows(
        ...     hwe_eas = agg.hardy_weinberg(agg.filter(dataset.pop == 'EAS', dataset.GT)))

    Notes
    -----
    This method returns a struct expression with the following fields:

    - `rExpectedHetFrequency` (:class:`hail.expr.types.TFloat64`) - Ratio of observed to
      expected heterozygote frequency.
    - `pHWE` (:class:`hail.expr.types.TFloat64`) - Hardy-Weinberg p-value.

    Hail computes the exact p-value with mid-p-value correction, i.e. the
    probability of a less-likely outcome plus one-half the probability of an
    equally-likely outcome. See this `document <LeveneHaldane.pdf>`__ for
    details on the Levene-Haldane distribution and references.

    Parameters
    ----------
    expr : :class:`hail.expr.expression.CallExpression`
        Call for which to compute Hardy-Weinberg statistics.

    Returns
    -------
    :class:`hail.expr.expression.StructExpression`
        Struct expression with fields `rExpectedHetFrequency` and `pHWE`.
    """
    t = TStruct(['rExpectedHetFrequency', 'pHWE'], [TFloat64(), TFloat64()])
    agg = _to_agg(expr)
    if not isinstance(agg._type, TCall):
        raise TypeError("aggregator 'hardy_weinberg' requires an expression of type 'TCall', found '{}'".format(
            agg._type.__class__))
    return _agg_func('hardyWeinberg', agg, t)


@typecheck(expr=oneof(expr_list, expr_set))
def explode(expr):
    """Explode an array or set expression to aggregate the elements of all records.

    Examples
    --------
    Compute the mean of all elements in fields `C1`, `C2`, and `C3`:

    .. doctest::

        >>> table1.aggregate(mean_c = agg.mean(agg.explode([table1.C1, table1.C2, table1.C3])))
        Struct(mean_c=24.8333333333)

    Compute the set of all observed elements in the `filters` field (``Set[String]``):

    .. doctest::

        >>> dataset.aggregate_rows(filters = agg.collect_as_set(agg.explode(dataset.filters)))
        Struct(filters=set([u'VQSRTrancheSNP99.80to99.90',
                            u'VQSRTrancheINDEL99.95to100.00',
                            u'VQSRTrancheINDEL99.00to99.50',
                            u'VQSRTrancheINDEL97.00to99.00',
                            u'VQSRTrancheSNP99.95to100.00',
                            u'VQSRTrancheSNP99.60to99.80',
                            u'VQSRTrancheINDEL99.50to99.90',
                            u'VQSRTrancheSNP99.90to99.95',
                            u'VQSRTrancheINDEL96.00to97.00']))

    Notes
    -----
    This method can be used with aggregator functions to aggregate the elements
    of collection types (:class:`hail.expr.types.TArray` and :class:`hail.expr.types.TSet`).

    The result of the :meth:`explode` and :meth:`filter` methods is an
    :class:`Aggregable` expression which can be used only in aggregator
    methods.

    Parameters
    ----------
    expr : :class:`hail.expr.expression.CollectionExpression`
        Expression of type :class:`hail.expr.types.TArray` or :class:`hail.expr.types.TSet`.

    Returns
    -------
    :class:`Aggregable`
        Aggregable expression.
    """
    agg = _to_agg(expr)
    uid = Env._get_uid()
    return Aggregable(LambdaClassMethod('flatMap', uid, agg._ast, Reference(uid)),
                      agg._type, agg._indices, agg._aggregations, agg._joins)


def filter(condition, expr):
    """Filter records according to a predicate.

    Examples
    --------
    Collect the `ID` field where `HT` >= 70:

    .. doctest::

        >>> table1.aggregate(high_ht = agg.collect(agg.filter(table1.HT >= 70, table1.ID)))
        Struct(high_ht=[2, 3])

    Notes
    -----
    This method can be used with aggregator functions to remove records from
    aggregation.

    The result of the :meth:`explode` and :meth:`filter` methods is an
    :class:`Aggregable` expression which can be used only in aggregator
    methods.

    Parameters
    ----------
    condition : :class:`hail.expr.expression.BooleanExpression` or function
        Filter expression, or a function to evaluate for each record.
    expr : :class:`hail.expr.expression.Expression`
        Expression to filter.

    Returns
    -------
    :class:`Aggregable`
        Aggregable expression.
    """

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
    """Compute inbreeding statistics on calls.

    Examples
    --------
    Compute inbreeding statistics per column:

    .. doctest::

        >>> dataset_result = dataset.annotate_cols(IB = agg.inbreeding(dataset.GT, dataset.variant_qc.AF))
        >>> dataset_result.cols_table().show()
        +----------------+--------------+-----------+------------+-----------------+-----------------+
        | s              |     IB.Fstat | IB.nTotal | IB.nCalled | IB.expectedHoms | IB.observedHoms |
        +----------------+--------------+-----------+------------+-----------------+-----------------+
        | String         |      Float64 |     Int64 |      Int64 |         Float64 |           Int64 |
        +----------------+--------------+-----------+------------+-----------------+-----------------+
        | C1046::HG02024 | -1.23867e-01 |       338 |        338 |     2.96180e+02 |             291 |
        | C1046::HG02025 |  2.02944e-02 |       339 |        339 |     2.97151e+02 |             298 |
        | C1046::HG02026 |  5.47269e-02 |       336 |        336 |     2.94742e+02 |             297 |
        | C1047::HG00731 | -1.89046e-02 |       337 |        337 |     2.95779e+02 |             295 |
        | C1047::HG00732 |  1.38718e-01 |       337 |        337 |     2.95202e+02 |             301 |
        | C1047::HG00733 |  3.50684e-01 |       338 |        338 |     2.96418e+02 |             311 |
        | C1048::HG02024 | -1.95603e-01 |       338 |        338 |     2.96180e+02 |             288 |
        | C1048::HG02025 |  2.02944e-02 |       339 |        339 |     2.97151e+02 |             298 |
        | C1048::HG02026 |  6.74296e-02 |       338 |        338 |     2.96180e+02 |             299 |
        | C1049::HG00731 | -1.00467e-02 |       337 |        337 |     2.95418e+02 |             295 |
        +----------------+--------------+-----------+------------+-----------------+-----------------+

    Notes
    -----

    ``E`` is total number of expected homozygous calls, given by the sum of
    ``1 - 2.0 * prior * (1 - prior)`` across records.

    ``O`` is the observed number of homozygous calls across records.

    ``N`` is the number of non-missing calls.

    ``F`` is the inbreeding coefficient, and is computed by: ``(O - E) / (N - E)``.

    This method returns a struct expression with five fields:

     - `Fstat` (:class:`hail.expr.types.TFloat64`): ``F``, the inbreeding coefficient.
     - `nTotal` (:class:`hail.expr.types.TInt64`): Total number of calls.
     - `nCalled` (:class:`hail.expr.types.TInt64`): ``N``, the number of non-missing calls.
     - `expectedHoms` (:class:`hail.expr.types.TFloat64`): ``E``, the expected number of homozygotes.
     - `observedHoms` (:class:`hail.expr.types.TInt64`): ``O``, the number of observed homozygotes.

    Parameters
    ----------
    expr : :class:`hail.expr.expression.CallExpression`
        Call expression.
    prior : :class:`hail.expr.expression.Float64Expression`
        Alternate allele frequency prior.

    Returns
    -------
    :class:`hail.expr.expression.StructExpression`
        Struct expression with fields `Fstat`, `nTotal`, `nCalled`, `expectedHoms`, `observedHoms`.
    """
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
    return construct_expr(ast, t, Indices(source=indices.source), aggregations.push(Aggregation(indices)), joins)


@typecheck(expr=oneof(Aggregable, expr_call), variant=expr_variant)
def call_stats(expr, variant):
    """Compute useful call statistics.

    Examples
    --------
    Compute call statistics per row:

    .. doctest::

        >>> dataset_result = dataset.annotate_rows(gt_stats = agg.call_stats(dataset.GT, dataset.v))
        >>> dataset_result.rows_table().select('v', 'gt_stats').show()
        +-----------------+--------------+----------------+-------------+--------------+
        | v               | gt_stats.AC  | gt_stats.AF    | gt_stats.AN | gt_stats.GC  |
        +-----------------+--------------+----------------+-------------+--------------+
        | Variant(GRCh37) | Array[Int32] | Array[Float64] |       Int32 | Array[Int32] |
        +-----------------+--------------+----------------+-------------+--------------+
        | 20:10019093:A:G | [111,89]     | [0.555,0.445]  |         200 | [36,39,25]   |
        | 20:10026348:A:G | [198,2]      | [0.99,0.01]    |         200 | [98,2,0]     |
        | 20:10026357:T:C | [166,34]     | [0.83,0.17]    |         200 | [68,30,2]    |
        | 20:10030188:T:A | [166,34]     | [0.83,0.17]    |         200 | [68,30,2]    |
        | 20:10030452:G:A | [170,30]     | [0.85,0.15]    |         200 | [71,28,1]    |
        | 20:10030508:T:C | [199,1]      | [0.995,0.005]  |         200 | [99,1,0]     |
        | 20:10030573:G:A | [198,2]      | [0.99,0.01]    |         200 | [98,2,0]     |
        | 20:10032413:T:G | [166,34]     | [0.83,0.17]    |         200 | [68,30,2]    |
        | 20:10036107:T:G | [187,13]     | [0.935,0.065]  |         200 | [87,13,0]    |
        | 20:10036141:C:T | [192,8]      | [0.96,0.04]    |         200 | [92,8,0]     |
        +-----------------+--------------+----------------+-------------+--------------+

    Notes
    -----
    This method is meaningful for computing call metrics per variant, but not
    especially meaningful for computing metrics per sample.

    This method returns a struct expression with four fields:

     - `AC` (:class:`hail.expr.types.TArray` of :class:`hail.expr.types.TInt32`) - Allele counts. One element
       for each allele, including the reference.
     - `AF` (:class:`hail.expr.types.TArray` of :class:`hail.expr.types.TFloat64`) - Allele frequencies. One
       element for each allele, including the reference.
     - `AN` (:class:`hail.expr.types.TInt32`) - Allele number. The total number of called
       alleles, or the number of non-missing calls * 2.
     - `GC` (:class:`hail.expr.types.TArray` of :class:`hail.expr.types.TInt32`) - Genotype counts. One element
       for each possible biallelic configuration (see
       :meth:`hail.expr.expression.VariantExpression.num_genotypes`).

    Parameters
    ----------
    expr : :class:`hail.expr.expression.CallExpression`
        Call.
    variant : :class:`hail.expr.expression.VariantExpression`
        Variant.

    Returns
    -------
    :class:`hail.expr.expression.StructExpression`
        Struct expression with fields `AC`, `AF`, `AN`, and `GC`.
    """
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
    return construct_expr(ast, t, Indices(source=indices.source), aggregations.push(Aggregation(indices)), joins)


def hist(expr, start, end, bins):
    """Compute binned counts of a numeric expression.

    Examples
    --------
    Compute a histogram of field `GQ`:

    .. doctest::

        >>> dataset.aggregate_entries(hist = agg.hist(dataset.GQ, 0, 100, 10))
        Struct(hist=Struct(binEdges=[0.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0],
                           binFrequencies=[2194L, 637L, 2450L, 1081L, 518L, 402L, 11168L, 1918L, 1379L, 11973L]),
                           nLess=0,
                           nGreater=0))

    Notes
    -----
    This method returns a struct expression with four fields:

     - `binEdges` (:class:`hail.expr.types.TArray` of :class:`hail.expr.types.TFloat64`): Bin edges. Bin `i`
       contains values in the left-inclusive, right-exclusive range
       ``[ binEdges[i], binEdges[i+1] )``.
     - `binFrequencies` (:class:`hail.expr.types.TArray` of :class:`hail.expr.types.TInt64`): Bin
       frequencies. The number of records found in each bin.
     - `nLess` (:class:`hail.expr.types.TInt64`): The number of records smaller than the start
       of the first bin.
     - `nGreater` (:class:`hail.expr.types.TInt64`): The number of records larger than the end
       of the last bin.

    Parameters
    ----------
    expr : :class:`hail.expr.expression.NumericExpression`
        Target numeric expression.
    start : :class:`hail.expr.expression.Float64Expression`
        Start of histogram range.
    end : :class:`hail.expr.expression.Float64Expression`
        End of histogram range.
    bins : :class:`hail.expr.expression.Int32Expression`
        Number of bins.

    Returns
    -------
    :class:`hail.expr.expression.StructExpression`
        Struct expression with fields `binEdges`, `binFrequencies`, `nLess`, and `nGreater`.
    """
    agg = _to_agg(expr)
    # FIXME check types
    t = TStruct(['binEdges', 'binFrequencies', 'nLess', 'nGreater'],
                [TArray(TFloat64()), TArray(TInt64()), TInt64(), TInt64()])
    return _agg_func('hist', agg, t, start, end, bins)
