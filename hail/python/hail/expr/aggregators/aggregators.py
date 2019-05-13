import difflib
from functools import wraps, update_wrapper

import hail as hl
from hail.expr.expressions import *
from hail.expr.types import *
from hail.expr.functions import rbind, float32, _quantile_from_cdf
from hail.ir import *
from hail.typecheck import *
from hail.utils import wrap_to_list
from hail.utils.java import Env


class AggregableChecker(TypeChecker):
    def __init__(self, coercer):
        self.coercer = coercer
        super(AggregableChecker, self).__init__()

    def expects(self):
        return self.coercer.expects()

    def format(self, arg):
        return self.coercer.format(arg)

    def check(self, x, caller, param):
        x = self.coercer.check(x, caller, param)
        if len(x._ir.search(lambda node: isinstance(node, BaseApplyAggOp))) == 0:
            raise ExpressionException(f"{caller} must be placed outside of an aggregation. See "
                                      "https://discuss.hail.is/t/breaking-change-redesign-of-aggregator-interface/701")
        return x


agg_expr = AggregableChecker


class AggFunc(object):
    def __init__(self):
        self._as_scan = False
        self._agg_bindings = set()

    def correct_prefix(self):
        return "scan" if self._as_scan else "agg"

    def incorrect_prefix(self):
        return "agg" if self._as_scan else "scan"

    def correct_plural(self):
        return "scans" if self._as_scan else "aggregations"

    def incorrect_plural(self):
        return "aggregations" if self._as_scan else "scans"

    def check_scan_agg_compatibility(self, caller, node):
        if self._as_scan != isinstance(node, ApplyScanOp):
            raise ExpressionException(
                "'{correct}.{caller}' cannot contain {incorrect}"
                    .format(correct=self.correct_prefix(),
                            caller=caller,
                            incorrect=self.incorrect_plural()))

    @typecheck_method(agg_op=str,
                      seq_op_args=sequenceof(expr_any),
                      ret_type=hail_type,
                      constructor_args=sequenceof(expr_any),
                      init_op_args=nullable(sequenceof(expr_any)))
    def __call__(self, agg_op, seq_op_args, ret_type, constructor_args=(), init_op_args=None):
        args = constructor_args if init_op_args is None else constructor_args + init_op_args
        indices, aggregations = unify_all(*seq_op_args, *args)
        if aggregations:
            raise ExpressionException('Cannot aggregate an already-aggregated expression')
        for a in seq_op_args + args:
            _check_agg_bindings(a, self._agg_bindings)

        if self._as_scan:
            ir = ApplyScanOp(agg_op,
                             [expr._ir for expr in constructor_args],
                             None if init_op_args is None else [expr._ir for expr in init_op_args],
                             [expr._ir for expr in seq_op_args])
            aggs = aggregations
        else:
            ir = ApplyAggOp(agg_op,
                            [expr._ir for expr in constructor_args],
                            None if init_op_args is None else [expr._ir for expr in init_op_args],
                            [expr._ir for expr in seq_op_args])
            aggs = aggregations.push(Aggregation(*seq_op_args, *args))
        return construct_expr(ir, ret_type, Indices(indices.source, set()), aggs)

    @typecheck_method(f=func_spec(1, expr_any),
                      array_agg_expr=expr_oneof(expr_array(), expr_set()))
    def explode(self, f, array_agg_expr):
        if len(array_agg_expr._ir.search(lambda n: isinstance(n, BaseApplyAggOp))) != 0:
            raise ExpressionException("'{}.explode' does not support an already-aggregated expression as the argument to 'collection'".format(self.correct_prefix()))
        _check_agg_bindings(array_agg_expr, self._agg_bindings)

        if isinstance(array_agg_expr.dtype, tset):
            array_agg_expr = hl.array(array_agg_expr)
        elt = array_agg_expr.dtype.element_type
        var = Env.get_uid()
        ref = construct_expr(Ref(var), elt, array_agg_expr._indices)
        self._agg_bindings.add(var)
        aggregated = f(ref)
        _check_agg_bindings(aggregated, self._agg_bindings)
        self._agg_bindings.remove(var)

        if len(aggregated._ir.search(lambda n: isinstance(n, BaseApplyAggOp))) == 0:
            raise ExpressionException("'{}.explode' must take mapping that contains aggregation expression.".format(self.correct_prefix()))

        indices, _ = unify_all(array_agg_expr, aggregated)
        aggregations = hl.utils.LinkedList(Aggregation)
        if not self._as_scan:
            aggregations = aggregations.push(Aggregation(array_agg_expr, aggregated))
        return construct_expr(AggExplode(array_agg_expr._ir, var, aggregated._ir, self._as_scan),
                              aggregated.dtype,
                              aggregated._indices,
                              aggregations)

    @typecheck_method(condition=expr_bool,
                      aggregation=agg_expr(expr_any))
    def filter(self, condition, aggregation):
        if len(condition._ir.search(lambda n: isinstance(n, BaseApplyAggOp))) != 0:
            raise ExpressionException(f"'hl.{self.correct_prefix()}.filter' does not "
                                      f"support an already-aggregated expression as the argument to 'condition'")
        if len(aggregation._ir.search(lambda n: isinstance(n, BaseApplyAggOp))) == 0:
            raise ExpressionException(f"'hl.{self.correct_prefix()}.filter' "
                                      f"must have aggregation in argument to 'aggregation'")

        _check_agg_bindings(condition, self._agg_bindings)
        _check_agg_bindings(aggregation, self._agg_bindings)
        unify_all(condition, aggregation)

        aggregations = hl.utils.LinkedList(Aggregation)
        if not self._as_scan:
            aggregations = aggregations.push(Aggregation(condition, aggregation))
        return construct_expr(AggFilter(condition._ir, aggregation._ir, self._as_scan),
                              aggregation.dtype,
                              aggregation._indices,
                              aggregations)

    def group_by(self, group, aggregation):
        if len(group._ir.search(lambda n: isinstance(n, BaseApplyAggOp))) != 0:
            raise ExpressionException(f"'hl.{self.correct_prefix()}.group_by' "
                                      f"does not support an already-aggregated expression as the argument to 'group'")
        if len(aggregation._ir.search(lambda n: isinstance(n, BaseApplyAggOp))) == 0:
            raise ExpressionException(f"'hl.{self.correct_prefix()}.group_by' "
                                      f"must have aggregation in argument to 'aggregation'")

        _check_agg_bindings(group, self._agg_bindings)
        _check_agg_bindings(aggregation, self._agg_bindings)
        unify_all(group, aggregation)

        aggregations = hl.utils.LinkedList(Aggregation)
        if not self._as_scan:
            aggregations = aggregations.push(Aggregation(aggregation))

        return construct_expr(AggGroupBy(group._ir, aggregation._ir, self._as_scan),
                              tdict(group.dtype, aggregation.dtype),
                              aggregation._indices,
                              aggregations)

    def array_agg(self, array, f):
        if len(array._ir.search(lambda n: isinstance(n, BaseApplyAggOp))) != 0:
            raise ExpressionException(f"'hl.{self.correct_prefix()}.array_agg' "
                                      f"does not support an already-aggregated expression as the argument to 'array'")
        _check_agg_bindings(array, self._agg_bindings)

        elt = array.dtype.element_type
        var = Env.get_uid()
        ref = construct_expr(Ref(var), elt, array._indices)
        self._agg_bindings.add(var)
        aggregated = f(ref)
        _check_agg_bindings(aggregated, self._agg_bindings)
        self._agg_bindings.remove(var)

        if len(aggregated._ir.search(lambda n: isinstance(n, BaseApplyAggOp))) == 0:
            raise ExpressionException(f"'hl.{self.correct_prefix()}.array_agg' "
                                      f"must take mapping that contains aggregation expression.")

        indices, _ = unify_all(array, aggregated)
        aggregations = hl.utils.LinkedList(Aggregation)
        if not self._as_scan:
            aggregations = aggregations.push(Aggregation(array, aggregated))
        return construct_expr(AggArrayPerElement(array._ir, var, 'unused', aggregated._ir, self._as_scan),
                              tarray(aggregated.dtype),
                              aggregated._indices,
                              aggregations)

_agg_func = AggFunc()


def _check_agg_bindings(expr, bindings):
    bound_references = {ref.name for ref in expr._ir.search(lambda ir: isinstance(ir, Ref) and not isinstance(ir, TopLevelReference))}
    free_variables = bound_references - expr._ir.bound_variables - bindings
    if free_variables:
        raise ExpressionException("dynamic variables created by 'hl.bind' or lambda methods like 'hl.map' may not be aggregated")


def approx_cdf(expr, k=100):
    """Produce a summary of the distribution of values.

    .. include: _templates/experimental.rst

    Notes
    -----
    This method returns a struct containing two arrays: `values` and `ranks`.
    The `values` array contains an ordered sample of values seen. The `ranks`
    array is one longer, and contains the approximate ranks for the
    corresponding values.

    These represent a summary of the CDF of the distribution of values. In
    particular, for any value `x = values(i)` in the summary, we estimate that
    there are `ranks(i)` values strictly less than `x`, and that there are
    `ranks(i+1)` values less than or equal to `x`. For any value `y` (not
    necessarily in the summary), we estimate CDF(y) to be `ranks(i)`, where `i`
    is such that `values(i-1) < y ≤ values(i)`.

    An alternative intuition is that the summary encodes a compressed
    approximation to the sorted list of values. For example, values=[0,2,5,6,9]
    and ranks=[0,3,4,5,8,10] represents the approximation [0,0,0,2,5,6,6,6,9,9],
    with the value `values(i)` occupying indices `ranks(i)` (inclusive) to
    `ranks(i+1)` (exclusive).

    Warning
    -------
    This is an approximate and nondeterministic method.

    Parameters
    ----------
    expr : :class:`.Expression`
        Expression to collect.
    k : :obj:`int`
        Parameter controlling the accuracy vs. memory usage tradeoff.

    Returns
    -------
    :class:`.StructExpression`
        Struct containing `values` and `ranks` arrays.
    """
    return _agg_func('ApproxCDF', [expr], tstruct(values=tarray(expr.dtype), ranks=tarray(tint64)), constructor_args=[k])


@typecheck(expr=expr_numeric, qs=expr_oneof(expr_numeric, expr_array(expr_numeric)), k=int)
def approx_quantiles(expr, qs, k=100) -> Expression:
    """Compute an array of approximate quantiles.

    .. include: _templates/experimental.rst

    Examples
    --------
    Estimate the median of the `HT` field.
    >>> table1.aggregate(hl.agg.approx_quantiles(table1.HT, 0.5)) # doctest: +NOTEST
    64

    Estimate the quartiles of the `HT` field.
    >>> table1.aggregate(hl.agg.approx_quantiles(table1.HT, [0, 0.25, 0.5, 0.75, 1])) # doctest: +NOTEST
    [50, 60, 64, 71, 86]

    Warning
    -------
    This is an approximate and nondeterministic method.

    Parameters
    ----------
    expr : :class:`.Expression`
        Expression to collect.
    qs : :class:`.NumericExpression` or :class:`.ArrayNumericExpression`
        Number or array of numbers between 0 and 1.
    k : :obj:`int`
        Parameter controlling the accuracy vs. memory usage tradeoff.

    Returns
    -------
    :class:`.NumericExpression` or :class:`.ArrayNumericExpression`
        If `qs` is a single number, returns the estimated quantile.
        If `qs` is an array, returns the array of estimated quantiles.
    """
    if isinstance(qs.dtype, tarray):
        return rbind(approx_cdf(expr, k), lambda cdf: qs.map(lambda q: _quantile_from_cdf(cdf, float32(q))))
    else:
        return rbind(approx_cdf(expr, k), lambda cdf: _quantile_from_cdf(cdf, qs))


@typecheck(expr=expr_any)
def collect(expr) -> ArrayExpression:
    """Collect records into an array.

    Examples
    --------
    Collect the `ID` field where `HT` is greater than 68:

    >>> table1.aggregate(hl.agg.filter(table1.HT > 68, hl.agg.collect(table1.ID)))
    [2, 3]

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
    expr : :class:`.Expression`
        Expression to collect.

    Returns
    -------
    :class:`.ArrayExpression`
        Array of all `expr` records.
    """
    return _agg_func('Collect', [expr], tarray(expr.dtype))


@typecheck(expr=expr_any)
def collect_as_set(expr) -> SetExpression:
    """Collect records into a set.

    Examples
    --------
    Collect the unique `ID` field where `HT` is greater than 68:

    >>> table1.aggregate(hl.agg.filter(table1.HT > 68, hl.agg.collect_as_set(table1.ID)))
    {2, 3}

    Warning
    -------
    Collecting a large number of items can cause out-of-memory exceptions.

    Parameters
    ----------
    expr : :class:`.Expression`
        Expression to collect.

    Returns
    -------
    :class:`.SetExpression`
        Set of unique `expr` records.
    """
    return _agg_func('CollectAsSet', [expr], tset(expr.dtype))


@typecheck()
def count() -> Int64Expression:
    """Count the number of records.

    Examples
    --------
    Group by the `SEX` field and count the number of rows in each category:

    >>> (table1.group_by(table1.SEX)
    ...        .aggregate(n=hl.agg.count())
    ...        .show())
    +-----+-------+
    | SEX |     n |
    +-----+-------+
    | str | int64 |
    +-----+-------+
    | "F" |     2 |
    | "M" |     2 |
    +-----+-------+

    Returns
    -------
    :class:`.Expression` of type :py:data:`.tint64`
        Total number of records.
    """
    return _agg_func('Count', [], tint64)


@typecheck(condition=expr_bool)
def count_where(condition) -> Int64Expression:
    """Count the number of records where a predicate is ``True``.

    Examples
    --------

    Count the number of individuals with `HT` greater than 68:

    >>> table1.aggregate(hl.agg.count_where(table1.HT > 68))
    2

    Parameters
    ----------
    condition : :class:`.BooleanExpression`
        Criteria for inclusion.

    Returns
    -------
    :class:`.Expression` of type :py:data:`.tint64`
        Total number of records where `condition` is ``True``.
    """

    return _agg_func('Sum', [hl.int64(condition)], tint64)


@typecheck(condition=expr_bool)
def any(condition) -> BooleanExpression:
    """Returns ``True`` if `condition` is ``True`` for any record.

    Examples
    --------

    >>> (table1.group_by(table1.SEX)
    ... .aggregate(any_over_70 = hl.agg.any(table1.HT > 70))
    ... .show())
    +-----+-------------+
    | SEX | any_over_70 |
    +-----+-------------+
    | str | bool        |
    +-----+-------------+
    | "F" | false       |
    | "M" | true        |
    +-----+-------------+

    Notes
    -----
    If there are no records to aggregate, the result is ``False``.

    Missing records are not considered. If every record is missing,
    the result is also ``False``.

    Parameters
    ----------
    condition : :class:`.BooleanExpression`
        Condition to test.

    Returns
    -------
    :class:`.BooleanExpression`
    """
    return count_where(condition) > 0


@typecheck(condition=expr_bool)
def all(condition) -> BooleanExpression:
    """Returns ``True`` if `condition` is ``True`` for every record.

    Examples
    --------

    >>> (table1.group_by(table1.SEX)
    ... .aggregate(all_under_70 = hl.agg.all(table1.HT < 70))
    ... .show())
    +-----+--------------+
    | SEX | all_under_70 |
    +-----+--------------+
    | str | bool         |
    +-----+--------------+
    | "F" | false        |
    | "M" | false        |
    +-----+--------------+

    Notes
    -----
    If there are no records to aggregate, the result is ``True``.

    Missing records are not considered. If every record is missing,
    the result is also ``True``.

    Parameters
    ----------
    condition : :class:`.BooleanExpression`
        Condition to test.

    Returns
    -------
    :class:`.BooleanExpression`
    """
    return count_where(~condition) == 0


@typecheck(expr=expr_any)
def counter(expr) -> DictExpression:
    """Count the occurrences of each unique record and return a dictionary.

    Examples
    --------
    Count the number of individuals for each unique `SEX` value:

    >>> table1.aggregate(hl.agg.counter(table1.SEX))  # doctest: +NOTEST
    {'M': 2L, 'F': 2L}

    Notes
    -----
    This aggregator method returns a dict expression whose key type is the
    same type as `expr` and whose value type is :class:`.Expression` of type :py:data:`.tint64`.
    This dict contains a key for each unique value of `expr`, and the value
    is the number of times that key was observed.

    Ensure that the result can be stored in memory on a single machine.

    Warning
    -------
    Using :meth:`counter` with a large number of unique items can cause
    out-of-memory exceptions.

    Parameters
    ----------
    expr : :class:`.Expression`
        Expression to count by key.

    Returns
    -------
    :class:`.DictExpression`
        Dictionary with the number of occurrences of each unique record.
    """
    return _agg_func('Counter', [expr], tdict(expr.dtype, tint64))


@typecheck(expr=expr_any,
           n=int,
           ordering=nullable(oneof(expr_any, func_spec(1, expr_any))))
def take(expr, n, ordering=None) -> ArrayExpression:
    """Take `n` records of `expr`, optionally ordered by `ordering`.

    Examples
    --------
    Take 3 elements of field `X`:

    >>> table1.aggregate(hl.agg.take(table1.X, 3))
    [5, 6, 7]

    Take the `ID` and `HT` fields, ordered by `HT` (descending):

    >>> table1.aggregate(hl.agg.take(hl.struct(ID=table1.ID, HT=table1.HT),
    ...                              3,
    ...                              ordering=-table1.HT))
    [Struct(ID=2, HT=72), Struct(ID=3, HT=70), Struct(ID=1, HT=65)]

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
    expr : :class:`.Expression`
        Expression to store.
    n : :class:`.Expression` of type :py:data:`.tint32`
        Number of records to take.
    ordering : :class:`.Expression` or function ((arg) -> :class:`.Expression`) or None
        Optional ordering on records.

    Returns
    -------
    :class:`.ArrayExpression`
        Array of up to `n` records of `expr`.

    """
    n = to_expr(n)
    if ordering is None:
        return _agg_func('Take', [expr], tarray(expr.dtype), [n])
    else:
        return _agg_func('TakeBy', [expr, ordering], tarray(expr.dtype), [n])


@typecheck(expr=expr_numeric)
def min(expr) -> NumericExpression:
    """Compute the minimum `expr`.

    Examples
    --------
    Compute the minimum value of `HT`:

    >>> table1.aggregate(hl.agg.min(table1.HT))
    60

    Notes
    -----
    This method returns the minimum non-missing value. If there are no
    non-missing values, then the result is missing.

    Parameters
    ----------
    expr : :class:`.NumericExpression`
        Numeric expression.

    Returns
    -------
    :class:`.NumericExpression`
        Minimum value of all `expr` records, same type as `expr`.
    """
    return _agg_func('Min', [expr], expr.dtype)


@typecheck(expr=expr_numeric)
def max(expr) -> NumericExpression:
    """Compute the maximum `expr`.

    Examples
    --------
    Compute the maximum value of `HT`:

    >>> table1.aggregate(hl.agg.max(table1.HT))
    72

    Notes
    -----
    This method returns the maximum non-missing value. If there are no
    non-missing values, then the result is missing.

    Parameters
    ----------
    expr : :class:`.NumericExpression`
        Numeric expression.

    Returns
    -------
    :class:`.NumericExpression`
        Maximum value of all `expr` records, same type as `expr`.
    """
    return _agg_func('Max', [expr], expr.dtype)


@typecheck(expr=expr_oneof(expr_int64, expr_float64))
def sum(expr):
    """Compute the sum of all records of `expr`.

    Examples
    --------
    Compute the sum of field `C1`:

    >>> table1.aggregate(hl.agg.sum(table1.C1))
    25

    Notes
    -----
    Missing values are ignored (treated as zero).

    If `expr` is an expression of type :py:data:`.tint32`, :py:data:`.tint64`,
    or :py:data:`.tbool`, then the result is an expression of type
    :py:data:`.tint64`. If `expr` is an expression of type :py:data:`.tfloat32`
    or :py:data:`.tfloat64`, then the result is an expression of type
    :py:data:`.tfloat64`.

    Warning
    -------
    Boolean values are cast to integers before computing the sum.

    Parameters
    ----------
    expr : :class:`.NumericExpression`
        Numeric expression.

    Returns
    -------
    :class:`.Expression` of type :py:data:`.tint64` or :py:data:`.tfloat64`
        Sum of records of `expr`.
    """
    return _agg_func('Sum', [expr], expr.dtype)


@typecheck(expr=expr_array(expr_oneof(expr_int64, expr_float64)))
def array_sum(expr) -> ArrayExpression:
    """Compute the coordinate-wise sum of all records of `expr`.

    Examples
    --------
    Compute the sum of `C1` and `C2`:

    >>> table1.aggregate(hl.agg.array_sum([table1.C1, table1.C2]))
    [25, 282]

    Notes
    ------
    All records must have the same length. Each coordinate is summed
    independently as described in :func:`sum`.

    Parameters
    ----------
    expr : :class:`.ArrayNumericExpression`

    Returns
    -------
    :class:`.ArrayExpression` with element type :py:data:`.tint64` or :py:data:`.tfloat64`
    """
    return _agg_func('Sum', [expr], expr.dtype)


@typecheck(expr=expr_float64)
def mean(expr) -> Float64Expression:
    """Compute the mean value of records of `expr`.

    Examples
    --------
    Compute the mean of field `HT`:

    >>> table1.aggregate(hl.agg.mean(table1.HT))
    66.75

    Notes
    -----
    Missing values are ignored.

    Parameters
    ----------
    expr : :class:`.NumericExpression`
        Numeric expression.

    Returns
    -------
    :class:`.Expression` of type :py:data:`.tfloat64`
        Mean value of records of `expr`.
    """
    return sum(expr)/count_where(hl.is_defined(expr))


@typecheck(expr=expr_float64)
def stats(expr) -> StructExpression:
    """Compute a number of useful statistics about `expr`.

    Examples
    --------
    Compute statistics about field `HT`:

    >>> table1.aggregate(hl.agg.stats(table1.HT))  #doctest: +SKIP
    Struct(mean=66.75, stdev=4.656984002549289, min=60.0, max=72.0, n=4, sum=267.0)

    Notes
    -----
    Computes a struct with the following fields:

    - `min` (:py:data:`.tfloat64`) - Minimum value.
    - `max` (:py:data:`.tfloat64`) - Maximum value.
    - `mean` (:py:data:`.tfloat64`) - Mean value,
    - `stdev` (:py:data:`.tfloat64`) - Standard deviation.
    - `n` (:py:data:`.tfloat64`) - Number of non-missing records.
    - `sum` (:py:data:`.tfloat64`) - Sum.

    Parameters
    ----------
    expr : :class:`.NumericExpression`
        Numeric expression.

    Returns
    -------
    :class:`.StructExpression`
        Struct expression with fields `mean`, `stdev`, `min`, `max`,
        `n`, and `sum`.
    """
    return _agg_func('Statistics', [expr], tstruct(mean=tfloat64,
                                                   stdev=tfloat64,
                                                   min=tfloat64,
                                                   max=tfloat64,
                                                   n=tint64,
                                                   sum=tfloat64))


@typecheck(expr=expr_oneof(expr_int64, expr_float64))
def product(expr):
    """Compute the product of all records of `expr`.

    Examples
    --------
    Compute the product of field `C1`:

    >>> table1.aggregate(hl.agg.product(table1.C1))
    440

    Notes
    -----
    Missing values are ignored (treated as one).

    If `expr` is an expression of type :py:data:`.tint32`, :py:data:`.tint64` or
    :py:data:`.tbool`, then the result is an expression of type
    :py:data:`.tint64`. If `expr` is an expression of type :py:data:`.tfloat32`
    or :py:data:`.tfloat64`, then the result is an expression of type
    :py:data:`.tfloat64`.

    Warning
    -------
    Boolean values are cast to integers before computing the product.

    Parameters
    ----------
    expr : :class:`.NumericExpression`
        Numeric expression.

    Returns
    -------
    :class:`.Expression` of type :py:data:`.tint64` or :py:data:`.tfloat64`
        Product of records of `expr`.
    """

    return _agg_func('Product', [expr], expr.dtype)


@typecheck(predicate=expr_bool)
def fraction(predicate) -> Float64Expression:
    """Compute the fraction of records where `predicate` is ``True``.

    Examples
    --------
    Compute the fraction of rows where `SEX` is "F" and `HT` > 65:

    >>> table1.aggregate(hl.agg.fraction((table1.SEX == 'F') & (table1.HT > 65)))
    0.25

    Notes
    -----
    Missing values for `predicate` are treated as ``False``.

    Parameters
    ----------
    predicate : :class:`.BooleanExpression`
        Boolean predicate.

    Returns
    -------
    :class:`.Expression` of type :py:data:`.tfloat64`
        Fraction of records where `predicate` is ``True``.
    """
    return _agg_func("Fraction", [predicate], tfloat64)


@typecheck(expr=expr_call)
def hardy_weinberg_test(expr) -> StructExpression:
    """Performs test of Hardy-Weinberg equilibrium.

    Examples
    --------
    Test each row of a dataset:

    >>> dataset_result = dataset.annotate_rows(hwe = hl.agg.hardy_weinberg_test(dataset.GT))

    Test each row on a sub-population:

    >>> dataset_result = dataset.annotate_rows(
    ...     hwe_eas = hl.agg.filter(dataset.pop == 'EAS',
    ...                             hl.agg.hardy_weinberg_test(dataset.GT)))

    Notes
    -----
    This method performs the test described in :func:`.functions.hardy_weinberg_test` based solely on
    the counts of homozygous reference, heterozygous, and homozygous variant calls.

    The resulting struct expression has two fields:

    - `het_freq_hwe` (:py:data:`.tfloat64`) - Expected frequency
      of heterozygous calls under Hardy-Weinberg equilibrium.

    - `p_value` (:py:data:`.tfloat64`) - p-value from test of Hardy-Weinberg
      equilibrium.

    Hail computes the exact p-value with mid-p-value correction, i.e. the
    probability of a less-likely outcome plus one-half the probability of an
    equally-likely outcome. See this `document <LeveneHaldane.pdf>`__ for
    details on the Levene-Haldane distribution and references.

    Warning
    -------
    Non-diploid calls (``ploidy != 2``) are ignored in the counts. While the
    counts are defined for multiallelic variants, this test is only statistically
    rigorous in the biallelic setting; use :func:`~hail.methods.split_multi`
    to split multiallelic variants beforehand.

    Parameters
    ----------
    expr : :class:`.CallExpression`
        Call to test for Hardy-Weinberg equilibrium.

    Returns
    -------
    :class:`.StructExpression`
        Struct expression with fields `het_freq_hwe` and `p_value`.
    """
    t = tstruct(het_freq_hwe=tfloat64, p_value=tfloat64)
    return _agg_func('HardyWeinberg', [expr], t)


@typecheck(f=func_spec(1, agg_expr(expr_any)), array_agg_expr=expr_oneof(expr_array(), expr_set()))
def explode(f, array_agg_expr) -> Expression:
    """Explode an array or set expression to aggregate the elements of all records.

    Examples
    --------
    Compute the mean of all elements in fields `C1`, `C2`, and `C3`:

    >>> table1.aggregate(hl.agg.explode(lambda elt: hl.agg.mean(elt), [table1.C1, table1.C2, table1.C3]))
    24.833333333333332

    Compute the set of all observed elements in the `filters` field (``Set[String]``):

    >>> dataset.aggregate_rows(hl.agg.explode(lambda elt: hl.agg.collect_as_set(elt), dataset.filters))
    {'VQSRTrancheINDEL97.00to99.00'}

    Notes
    -----
    This method can be used with aggregator functions to aggregate the elements
    of collection types (:class:`.tarray` and :class:`.tset`).

    Parameters
    ----------
    f : Function from :class:`.Expression` to :class:`.Expression`
        Aggregation function to apply to each element of the exploded array.
    array_agg_expr : :class:`.CollectionExpression`
        Expression of type :class:`.tarray` or :class:`.tset`.

    Returns
    -------
    :class:`.Expression`
        Aggregation expression.
    """
    return _agg_func.explode(f, array_agg_expr)


@typecheck(condition=expr_bool, aggregation=agg_expr(expr_any))
def filter(condition,  aggregation) -> Expression:
    """Filter records according to a predicate.

    Examples
    --------
    Collect the `ID` field where `HT` >= 70:

    >>> table1.aggregate(hl.agg.filter(table1.HT >= 70, hl.agg.collect(table1.ID)))
    [2, 3]

    Notes
    -----
    This method can be used with aggregator functions to remove records from
    aggregation.

    Parameters
    ----------
    condition : :class:`.BooleanExpression`
        Filter expression.
    aggregation : :class:`.Expression`
        Aggregation expression to filter.

    Returns
    -------
    :class:`.Aggregable`
        Aggregable expression.
    """

    return _agg_func.filter(condition, aggregation)


@typecheck(expr=expr_call, prior=expr_float64)
def inbreeding(expr, prior) -> StructExpression:
    """Compute inbreeding statistics on calls.

    Examples
    --------
    Compute inbreeding statistics per column:

    >>> dataset_result = dataset.annotate_cols(IB = hl.agg.inbreeding(dataset.GT, dataset.variant_qc.AF[1]))
    >>> dataset_result.IB.show()
    +------------------+-----------+-------------+------------------+------------------+
    | s                | IB.f_stat | IB.n_called | IB.expected_homs | IB.observed_homs |
    +------------------+-----------+-------------+------------------+------------------+
    | str              |   float64 |       int64 |          float64 |            int64 |
    +------------------+-----------+-------------+------------------+------------------+
    | "C1046::HG02024" |  3.88e-01 |          12 |         1.04e+01 |               11 |
    | "C1046::HG02025" |  3.99e-01 |          13 |         1.13e+01 |               12 |
    | "C1046::HG02026" |  3.88e-01 |          12 |         1.04e+01 |               11 |
    | "C1047::HG00731" | -1.31e+00 |          12 |         1.07e+01 |                9 |
    | "C1047::HG00732" |  1.00e+00 |          12 |         1.04e+01 |               12 |
    | "C1047::HG00733" | -8.04e-01 |          13 |         1.13e+01 |               10 |
    | "C1048::HG02024" |  3.88e-01 |          12 |         1.04e+01 |               11 |
    | "C1048::HG02025" |  3.99e-01 |          13 |         1.13e+01 |               12 |
    | "C1048::HG02026" |  1.00e+00 |          12 |         1.04e+01 |               12 |
    | "C1049::HG00731" | -1.41e+00 |          13 |         1.13e+01 |                9 |
    +------------------+-----------+-------------+------------------+------------------+
    showing top 10 rows

    Notes
    -----

    ``E`` is total number of expected homozygous calls, given by the sum of
    ``1 - 2.0 * prior * (1 - prior)`` across records.

    ``O`` is the observed number of homozygous calls across records.

    ``N`` is the number of non-missing calls.

    ``F`` is the inbreeding coefficient, and is computed by: ``(O - E) / (N - E)``.

    This method returns a struct expression with four fields:

     - `f_stat` (:py:data:`.tfloat64`): ``F``, the inbreeding coefficient.
     - `n_called` (:py:data:`.tint64`): ``N``, the number of non-missing calls.
     - `expected_homs` (:py:data:`.tfloat64`): ``E``, the expected number of homozygotes.
     - `observed_homs` (:py:data:`.tint64`): ``O``, the number of observed homozygotes.

    Parameters
    ----------
    expr : :class:`.CallExpression`
        Call expression.
    prior : :class:`.Expression` of type :py:data:`.tfloat64`
        Alternate allele frequency prior.

    Returns
    -------
    :class:`.StructExpression`
        Struct expression with fields `f_stat`, `n_called`, `expected_homs`, `observed_homs`.
    """
    t = tstruct(f_stat=tfloat64,
                n_called=tint64,
                expected_homs=tfloat64,
                observed_homs=tint64)
    return _agg_func('Inbreeding', [expr, prior], t)


@typecheck(call=expr_call, alleles=expr_array(expr_str))
def call_stats(call, alleles) -> StructExpression:
    """Compute useful call statistics.

    Examples
    --------
    Compute call statistics per row:

    >>> dataset_result = dataset.annotate_rows(gt_stats = hl.agg.call_stats(dataset.GT, dataset.alleles))
    >>> dataset_result.rows().key_by('locus').select('gt_stats').show()
    +---------------+--------------+---------------------+-------------+
    | locus         | gt_stats.AC  | gt_stats.AF         | gt_stats.AN |
    +---------------+--------------+---------------------+-------------+
    | locus<GRCh37> | array<int32> | array<float64>      |       int32 |
    +---------------+--------------+---------------------+-------------+
    | 20:12990057   | [148,52]     | [7.40e-01,2.60e-01] |         200 |
    | 20:13029862   | [0,198]      | [0.00e+00,1.00e+00] |         198 |
    | 20:13074235   | [13,187]     | [6.50e-02,9.35e-01] |         200 |
    | 20:13140720   | [194,6]      | [9.70e-01,3.00e-02] |         200 |
    | 20:13695498   | [175,25]     | [8.75e-01,1.25e-01] |         200 |
    | 20:13714384   | [199,1]      | [9.95e-01,5.00e-03] |         200 |
    | 20:13765944   | [132,2]      | [9.85e-01,1.49e-02] |         134 |
    | 20:13765954   | [180,2]      | [9.89e-01,1.10e-02] |         182 |
    | 20:13845987   | [2,198]      | [1.00e-02,9.90e-01] |         200 |
    | 20:16223957   | [145,45]     | [7.63e-01,2.37e-01] |         190 |
    +---------------+--------------+---------------------+-------------+
    <BLANKLINE>
    +---------------------------+
    | gt_stats.homozygote_count |
    +---------------------------+
    | array<int32>              |
    +---------------------------+
    | [57,9]                    |
    | [0,99]                    |
    | [1,88]                    |
    | [95,1]                    |
    | [75,0]                    |
    | [99,0]                    |
    | [65,0]                    |
    | [89,0]                    |
    | [0,98]                    |
    | [64,14]                   |
    +---------------------------+
    showing top 10 rows
    <BLANKLINE>

    Notes
    -----
    This method is meaningful for computing call metrics per variant, but not
    especially meaningful for computing metrics per sample.

    This method returns a struct expression with three fields:

     - `AC` (:class:`.tarray` of :py:data:`.tint32`) - Allele counts. One element
       for each allele, including the reference.
     - `AF` (:class:`.tarray` of :py:data:`.tfloat64`) - Allele frequencies. One
       element for each allele, including the reference.
     - `AN` (:py:data:`.tint32`) - Allele number. The total number of called
       alleles, or the number of non-missing calls * 2.
     - `homozygote_count` (:class:`.tarray` of :py:data:`.tint32`) - Homozygote
       genotype counts for each allele, including the reference. Only **diploid**
       genotype calls are counted.

    Parameters
    ----------
    call : :class:`.CallExpression`
    alleles : :class:`.ArrayStringExpression`
        Variant alleles.

    Returns
    -------
    :class:`.StructExpression`
        Struct expression with fields `AC`, `AF`, `AN`, and `homozygote_count`.
    """
    n_alleles = hl.len(alleles)
    t = tstruct(AC=tarray(tint32),
                AF=tarray(tfloat64),
                AN=tint32,
                homozygote_count=tarray(tint32))

    return _agg_func('CallStats', [call], t, [], init_op_args=[n_alleles])


@typecheck(expr=expr_float64, start=expr_float64, end=expr_float64, bins=expr_int32)
def hist(expr, start, end, bins) -> StructExpression:
    """Compute binned counts of a numeric expression.

    Examples
    --------
    Compute a histogram of field `GQ`:

    >>> dataset.aggregate_entries(hl.agg.hist(dataset.GQ, 0, 100, 10))  # doctest: +NOTEST
    Struct(bin_edges=[0.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0],
           bin_freq=[2194L, 637L, 2450L, 1081L, 518L, 402L, 11168L, 1918L, 1379L, 11973L]),
           n_smaller=0,
           n_greater=0)

    Notes
    -----
    This method returns a struct expression with four fields:

     - `bin_edges` (:class:`.tarray` of :py:data:`.tfloat64`): Bin edges. Bin `i`
       contains values in the left-inclusive, right-exclusive range
       ``[ bin_edges[i], bin_edges[i+1] )``.
     - `bin_freq` (:class:`.tarray` of :py:data:`.tint64`): Bin
       frequencies. The number of records found in each bin.
     - `n_smaller` (:py:data:`.tint64`): The number of records smaller than the start
       of the first bin.
     - `n_larger` (:py:data:`.tint64`): The number of records larger than the end
       of the last bin.

    Parameters
    ----------
    expr : :class:`.NumericExpression`
        Target numeric expression.
    start : :obj:`int` or :obj:`float`
        Start of histogram range.
    end : :obj:`int` or :obj:`float`
        End of histogram range.
    bins : :obj:`int`
        Number of bins.

    Returns
    -------
    :class:`.StructExpression`
        Struct expression with fields `bin_edges`, `bin_freq`, `n_smaller`, and `n_larger`.
    """
    t = tstruct(bin_edges=tarray(tfloat64),
                bin_freq=tarray(tint64),
                n_smaller=tint64,
                n_larger=tint64)
    return _agg_func('Histogram', [expr], t, constructor_args=[start, end, bins])


@typecheck(x=expr_float64, y=expr_float64, label=nullable(oneof(expr_str, expr_array(expr_str))), n_divisions=int)
def downsample(x, y, label=None, n_divisions=500) -> ArrayExpression:
    """Downsample (x, y) coordinate datapoints.

    .. include: _templates/experimental.rst

    Parameters
    ---------
    x : :class:`.NumericExpression`
        X-values to be downsampled.
    y : :class:`.NumericExpression`
        Y-values to be downsampled.
    label : :class:`.StringExpression` or :class:`.ArrayExpression`
        Additional data for each (x, y) coordinate. Can pass in multiple fields in an :class:`.ArrayExpression`.
    n_divisions : :obj:`int`
        Factor by which to downsample (default value = 500). A lower input results in fewer output datapoints.

    Returns
    -------
    :class:`.ArrayExpression`
        Expression for downsampled coordinate points (x, y). The element type of the array is
        :py:data:`.ttuple` of :py:data:`.tfloat64`, :py:data:`.tfloat64`, and :py:data:`.tarray` of :py:data:`.tstring`
    """
    if label is None:
        label = hl.null(hl.tarray(hl.tstr))
    elif isinstance(label, StringExpression):
        label = hl.array([label])
    return _agg_func('Downsample', [x, y, label], tarray(ttuple(tfloat64, tfloat64, tarray(tstr))),
                     constructor_args=[n_divisions])


@typecheck(gp=expr_array(expr_float64))
def info_score(gp) -> StructExpression:
    r"""Compute the IMPUTE information score.

    Examples
    --------
    Calculate the info score per variant:

    >>> gen_mt = hl.import_gen('data/example.gen', sample_file='data/example.sample')
    >>> gen_mt = gen_mt.annotate_rows(info_score = hl.agg.info_score(gen_mt.GP))

    Calculate group-specific info scores per variant:

    >>> gen_mt = hl.import_gen('data/example.gen', sample_file='data/example.sample')
    >>> gen_mt = gen_mt.annotate_cols(is_case = hl.rand_bool(0.5))
    >>> gen_mt = gen_mt.annotate_rows(info_score = hl.agg.group_by(gen_mt.is_case, hl.agg.info_score(gen_mt.GP)))

    Notes
    -----
    The result of :func:`.info_score` is a struct with two fields:

        - `score` (``float64``) -- Info score.
        - `n_included` (``int32``) -- Number of non-missing samples included in the
          calculation.

    We implemented the IMPUTE info measure as described in the supplementary
    information from `Marchini & Howie. Genotype imputation for genome-wide
    association studies. Nature Reviews Genetics (2010)
    <http://www.nature.com/nrg/journal/v11/n7/extref/nrg2796-s3.pdf>`__. To
    calculate the info score :math:`I_{A}` for one SNP:

    .. math::

        I_{A} =
        \begin{cases}
        1 - \frac{\sum_{i=1}^{N}(f_{i} - e_{i}^2)}{2N\hat{\theta}(1 - \hat{\theta})} & \text{when } \hat{\theta} \in (0, 1) \\
        1 & \text{when } \hat{\theta} = 0, \hat{\theta} = 1\\
        \end{cases}

    - :math:`N` is the number of samples with imputed genotype probabilities
      [:math:`p_{ik} = P(G_{i} = k)` where :math:`k \in \{0, 1, 2\}`]
    - :math:`e_{i} = p_{i1} + 2p_{i2}` is the expected genotype per sample
    - :math:`f_{i} = p_{i1} + 4p_{i2}`
    - :math:`\hat{\theta} = \frac{\sum_{i=1}^{N}e_{i}}{2N}` is the MLE for the
      population minor allele frequency

    Hail will not generate identical results to `QCTOOL
    <http://www.well.ox.ac.uk/~gav/qctool/#overview>`__ for the following
    reasons:

    - Hail automatically removes genotype probability distributions that do not
      meet certain requirements on data import with :func:`.import_gen` and
      :func:`.import_bgen`.
    - Hail does not use the population frequency to impute genotype
      probabilities when a genotype probability distribution has been set to
      missing.
    - Hail calculates the same statistic for sex chromosomes as autosomes while
      QCTOOL incorporates sex information.
    - The floating point number Hail stores for each genotype probability is
      slightly different than the original data due to rounding and
      normalization of probabilities.

    Warning
    -------
    - The info score Hail reports will be extremely different from QCTOOL when
      a SNP has a high missing rate.
    - If the `gp` array must contain 3 elements, and its elements may not be
      missing.
    - If the genotype data was not imported using the :func:`.import_gen` or
      :func:`.import_bgen` functions, then the results for all variants will be
      ``score = NA`` and ``n_included = 0``.
    - It only makes semantic sense to compute the info score per variant. While
      the aggregator will run in any context if its arguments are the right
      type, the results are only meaningful in a narrow context.

    Parameters
    ----------
    gp : :class:`.ArrayNumericExpression`
        Genotype probability array. Must have 3 elements, all of which must be
        defined.

    Returns
    -------
    :class:`.StructExpression`
        Struct with fields `score` and `n_included`.
    """
    t = hl.tstruct(score=hl.tfloat64, n_included=hl.tint32)
    return _agg_func('InfoScore', [gp], t)


@typecheck(y=expr_float64,
           x=oneof(expr_float64, sequenceof(expr_float64)),
           nested_dim=int,
           weight=nullable(expr_float64))
def linreg(y, x, nested_dim=1, weight=None) -> StructExpression:
    """Compute multivariate linear regression statistics.

    Examples
    --------
    Regress HT against an intercept (1), SEX, and C1:

    >>> table1.aggregate(hl.agg.linreg(table1.HT, [1, table1.SEX == 'F', table1.C1]))  # doctest: +NOTEST
    Struct(beta=[88.50000000000014, 81.50000000000057, -10.000000000000068],
           standard_error=[14.430869689661844, 59.70552738231206, 7.000000000000016],
           t_stat=[6.132686518775844, 1.365032746099571, -1.428571428571435],
           p_value=[0.10290201427537926, 0.40250974549499974, 0.3888002244284281],
           multiple_standard_error=4.949747468305833,
           multiple_r_squared=0.7175792507204611,
           adjusted_r_squared=0.1527377521613834,
           f_stat=1.2704081632653061,
           multiple_p_value=0.5314327326007864,
           n=4)

    Regress blood pressure against an intercept (1), genotype, age, and
    the interaction of genotype and age:

    >>> ds_ann = ds.annotate_rows(linreg = 
    ...     hl.agg.linreg(ds.pheno.blood_pressure,
    ...                   [1,
    ...                    ds.GT.n_alt_alleles(),
    ...                    ds.pheno.age,
    ...                    ds.GT.n_alt_alleles() * ds.pheno.age]))

    Warning
    -------
    As in the example, the intercept covariate ``1`` must be included
    **explicitly** if desired.

    Notes
    -----
    In relation to
    `lm.summary <https://stat.ethz.ch/R-manual/R-devel/library/stats/html/summary.lm.html>`__
    in R, ``linreg(y, x = [1, mt.x1, mt.x2])`` computes
    ``summary(lm(y ~ x1 + x2))`` and
    ``linreg(y, x = [mt.x1, mt.x2], nested_dim=0)`` computes
    ``summary(lm(y ~ x1 + x2 - 1))``.

    More generally, `nested_dim` defines the number of effects to fit in the
    nested (null) model, with the effects on the remaining covariates fixed
    to zero.

    The returned struct has ten fields:
     - `beta` (:class:`.tarray` of :py:data:`.tfloat64`):
       Estimated regression coefficient for each covariate.
     - `standard_error` (:class:`.tarray` of :py:data:`.tfloat64`):
       Estimated standard error for each covariate.
     - `t_stat` (:class:`.tarray` of :py:data:`.tfloat64`):
       t-statistic for each covariate.
     - `p_value` (:class:`.tarray` of :py:data:`.tfloat64`):
       p-value for each covariate.
     - `multiple_standard_error` (:py:data:`.tfloat64`):
       Estimated standard deviation of the random error.
     - `multiple_r_squared` (:py:data:`.tfloat64`):
       Coefficient of determination for nested models.
     - `adjusted_r_squared` (:py:data:`.tfloat64`):
       Adjusted `multiple_r_squared` taking into account degrees of
       freedom.
     - `f_stat` (:py:data:`.tfloat64`):
       F-statistic for nested models.
     - `multiple_p_value` (:py:data:`.tfloat64`):
       p-value for the
       `F-test <https://en.wikipedia.org/wiki/F-test#Regression_problems>`__ of
       nested models.
     - `n` (:py:data:`.tint64`):
       Number of samples included in the regression. A sample is included if and
       only if `y`, all elements of `x`, and `weight` (if set) are non-missing.

    All but the last field are missing if `n` is less than or equal to the
    number of covariates or if the covariates are linearly dependent.

    If set, the `weight` parameter generalizes the model to `weighted least
    squares <https://en.wikipedia.org/wiki/Weighted_least_squares>`__, useful
    for heteroscedastic (diagonal but non-constant) variance.

    Warning
    -------
    If any weight is negative, the resulting statistics will be ``nan``.

    Parameters
    ----------
    y : :class:`.Float64Expression`
        Response (dependent variable).
    x : :class:`.Float64Expression` or :obj:`list` of :class:`.Float64Expression`
        Covariates (independent variables).
    nested_dim : :obj:`int`
        The null model includes the first `nested_dim` covariates.
        Must be between 0 and `k` (the length of `x`).
    weight : :class:`.Float64Expression`, optional
        Non-negative weight for weighted least squares.

    Returns
    -------
    :class:`.StructExpression`
        Struct of regression results.
    """
    x = wrap_to_list(x)
    if len(x) == 0:
        raise ValueError("linreg: must have at least one covariate in `x`")

    hl.methods.statgen._warn_if_no_intercept('linreg', x)

    if weight is None:
        return _linreg(y, x, nested_dim)
    else:
        return _linreg(hl.sqrt(weight) * y,
                       [hl.sqrt(weight) * xi for xi in x],
                       nested_dim)


def _linreg(y, x, nested_dim):
    k = len(x)
    k0 = nested_dim
    if k0 < 0 or k0 > k:
        raise ValueError("linreg: `nested_dim` must be between 0 and the number "
                         f"of covariates ({k}), inclusive")

    t = hl.tstruct(beta=hl.tarray(hl.tfloat64),
                   standard_error=hl.tarray(hl.tfloat64),
                   t_stat=hl.tarray(hl.tfloat64),
                   p_value=hl.tarray(hl.tfloat64),
                   multiple_standard_error=hl.tfloat64,
                   multiple_r_squared=hl.tfloat64,
                   adjusted_r_squared=hl.tfloat64,
                   f_stat=hl.tfloat64,
                   multiple_p_value=hl.tfloat64,
                   n=hl.tint64)

    x = hl.array(x)
    k = hl.int32(k)
    k0 = hl.int32(k0)

    return _agg_func('LinearRegression', [y, x], t, [k, k0])

@typecheck(x=expr_float64, y=expr_float64)
def corr(x, y) -> Float64Expression:
    """Computes the
    `Pearson correlation coefficient <https://en.wikipedia.org/wiki/Pearson_correlation_coefficient>`__
    between `x` and `y`.

    Examples
    --------
    >>> ds.aggregate_cols(hl.agg.corr(ds.pheno.age, ds.pheno.blood_pressure))  # doctest: +NOTEST
    0.16592876044845484

    Notes
    -----
    Only records where both `x` and `y` are non-missing will be included in the
    calculation.

    In the case that there are no non-missing pairs, the result will be missing.

    See Also
    --------
    :func:`linreg`

    Parameters
    ----------
    x : :class:`.Expression` of type ``tfloat64``
    y : :class:`.Expression` of type ``tfloat64``

    Returns
    -------
    :class:`.Float64Expression`
    """
    return _agg_func('PearsonCorrelation', [x, y], tfloat64)

@typecheck(group=expr_any,
           agg_expr=agg_expr(expr_any))
def group_by(group, agg_expr) -> DictExpression:
    """Compute aggregation statistics stratified by one or more groups.

    .. include:: _templates/experimental.rst

    Examples
    --------
    Compute linear regression statistics stratified by SEX:

    >>> table1.aggregate(hl.agg.group_by(table1.SEX,
    ...                                  hl.agg.linreg(table1.HT, table1.C1, nested_dim=0)))  # doctest: +NOTEST
    {
    'F': Struct(beta=[6.153846153846154],
                standard_error=[0.7692307692307685],
                t_stat=[8.000000000000009],
                p_value=[0.07916684832113098],
                multiple_standard_error=11.4354374979373,
                multiple_r_squared=0.9846153846153847,
                adjusted_r_squared=0.9692307692307693,
                f_stat=64.00000000000014,
                multiple_p_value=0.07916684832113098,
                n=2),
    'M': Struct(beta=[34.25],
                standard_error=[1.75],
                t_stat=[19.571428571428573],
                p_value=[0.03249975499062629],
                multiple_standard_error=4.949747468305833,
                multiple_r_squared=0.9973961101073441,
                adjusted_r_squared=0.9947922202146882,
                f_stat=383.0408163265306,
                multiple_p_value=0.03249975499062629,
                n=2)
    }

    Compute call statistics stratified by population group and case status:

    >>> ann = ds.annotate_rows(call_stats=hl.agg.group_by(hl.struct(pop=ds.pop, is_case=ds.is_case),
    ...                                                   hl.agg.call_stats(ds.GT, ds.alleles)))

    Parameters
    ----------
    group : :class:`.Expression` or :obj:`list` of :class:`.Expression`
        Group to stratify the result by.
    agg_expr : :class:`.Expression`
        Aggregation or scan expression to compute per grouping.

    Returns
    -------
    :class:`.DictExpression`
        Dictionary where the keys are `group` and the values are the result of computing
        `agg_expr` for each unique value of `group`.
    """

    return _agg_func.group_by(group, agg_expr)

@typecheck(expr=expr_any)
def _prev_nonnull(expr) -> ArrayExpression:
    wrap = expr.dtype in {tint32, tint64, tfloat32, tfloat64, tbool, tcall}
    if wrap:
        expr = hl.tuple([expr])
    r = _agg_func('PrevNonnull', [expr], expr.dtype, [])
    if wrap:
        r = r[0]
    return r

@typecheck(f=func_spec(1, expr_any),
           array=expr_array())
def array_agg(f, array):
    """Aggregate an array element-wise using a user-specified aggregation function.

    Examples
    --------
    Start with a range table with an array of random boolean values:

    >>> ht = hl.utils.range_table(100)
    >>> ht = ht.annotate(arr = hl.range(0, 5).map(lambda _: hl.rand_bool(0.5)))

    Aggregate to compute the fraction ``True`` per element:

    >>> ht.aggregate(hl.agg.array_agg(lambda element: hl.agg.fraction(element), ht.arr))  # doctest: +NOTEST
    [0.54, 0.55, 0.46, 0.52, 0.48]

    Notes
    -----
    This function requires that all values of `array` have the same length. If
    two values have different lengths, then an exception will be thrown.

    The `f` argument should be a function taking one argument, an expression of
    the element type of `array`, and returning an expression including
    aggregation(s). The type of the aggregated expression returned by
    :func:`array_agg` is an array of elements of the return type of `f`.

    Parameters
    ----------
    f :
        Aggregation function to apply to each element of the exploded array.
    array : :class:`.ArrayExpression`
        Array to aggregate.

    Returns
    -------
    :class:`.ArrayExpression`
    """
    return _agg_func.array_agg(array, f)


class ScanFunctions(object):

    def __init__(self, scope):
        self._functions = {name: self._scan_decorator(f) for name, f in scope.items()}

    def _scan_decorator(self, f):
        @wraps(f)
        def wrapper(*args, **kwargs):
            func = getattr(f, '__wrapped__')
            af = func.__globals__['_agg_func']
            as_scan = getattr(af, '_as_scan')
            setattr(af, '_as_scan', True)
            try:
                res = f(*args, **kwargs)
            except Exception as e:
                setattr(af, '_as_scan', as_scan)
                raise e
            setattr(af, '_as_scan', as_scan)
            return res
        update_wrapper(wrapper, f)
        return wrapper

    def __getattr__(self, field):
        if field in self._functions:
            return self._functions[field]
        else:
            field_matches = difflib.get_close_matches(field, self._functions.keys(), n=5)
            raise AttributeError("hl.scan.{} does not exist. Did you mean:\n    {}".format(
                field,
                "\n    ".join(field_matches)))
