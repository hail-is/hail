import builtins
import functools
from typing import *

import hail as hl
from hail.expr.expressions import *
from hail.expr.expressions.expression_typecheck import *
from hail.expr.types import *
from hail.genetics.reference_genome import reference_genome_type, ReferenceGenome
from hail.ir import *
from hail.typecheck import *
from hail.utils.java import Env
from hail.utils.misc import plural

import numpy as np

Coll_T = TypeVar('Collection_T', ArrayExpression, SetExpression)
Num_T = TypeVar('Numeric_T', Int32Expression, Int64Expression, Float32Expression, Float64Expression)


def _func(name, ret_type, *args):
    indices, aggregations = unify_all(*args)
    return construct_expr(Apply(name, *(a._ir for a in args)), ret_type, indices, aggregations)


def _seeded_func(name, ret_type, seed, *args):
    seed = seed if seed is not None else Env.next_seed()
    indices, aggregations = unify_all(*args)
    return construct_expr(ApplySeeded(name, seed, *(a._ir for a in args)), ret_type, indices, aggregations)


@typecheck(a=expr_array(), x=expr_any)
def _lower_bound(a, x):
    if a.dtype.element_type != x.dtype:
        raise TypeError(f"_lower_bound: incompatible types: {a.dtype}, {x.dtype}")
    indices, aggregations = unify_all(a, x)
    return construct_expr(LowerBoundOnOrderedCollection(a._ir, x._ir, on_key=False), tint32, indices, aggregations)


@typecheck(cdf=expr_struct(), q=expr_oneof(expr_float32, expr_float64))
def _quantile_from_cdf(cdf, q):
    n = cdf.ranks[cdf.ranks.length() - 1]
    pos = int64(q*n) + 1
    idx = (switch(q)
            .when(0.0, 0)
            .when(1.0, cdf.values.length() - 1)
            .default(_lower_bound(cdf.ranks, pos) - 1))
    return cdf.values[idx]


@typecheck(t=hail_type)
def null(t: Union[HailType, str]):
    """Creates an expression representing a missing value of a specified type.

    Examples
    --------

    >>> hl.eval(hl.null(hl.tarray(hl.tstr)))
    None

    >>> hl.eval(hl.null('array<str>'))
    None

    Notes
    -----
    This method is useful for constructing an expression that includes missing
    values, since :obj:`None` cannot be interpreted as an expression.

    Parameters
    ----------
    t : :obj:`str` or :class:`.HailType`
        Type of the missing expression.

    Returns
    -------
    :class:`.Expression`
        A missing expression of type `t`.
    """
    return construct_expr(NA(t), t)


@typecheck(x=anytype, dtype=nullable(hail_type))
def literal(x: Any, dtype: Optional[Union[HailType, str]] = None):
    """Captures and broadcasts a Python variable or object as an expression.

    Examples
    --------

    >>> table = hl.utils.range_table(8)
    >>> greetings = hl.literal({1: 'Good morning', 4: 'Good afternoon', 6 : 'Good evening'})
    >>> table.annotate(greeting = greetings.get(table.idx)).show()
    +-------+------------------+
    |   idx | greeting         |
    +-------+------------------+
    | int32 | str              |
    +-------+------------------+
    |     0 | NA               |
    |     1 | "Good morning"   |
    |     2 | NA               |
    |     3 | NA               |
    |     4 | "Good afternoon" |
    |     5 | NA               |
    |     6 | "Good evening"   |
    |     7 | NA               |
    +-------+------------------+

    Notes
    -----
    Use this function to capture large Python objects for use in expressions. This
    function provides an alternative to adding an object as a global annotation on a
    :class:`.Table` or :class:`.MatrixTable`.

    Parameters
    ----------
    x
        Object to capture and broadcast as an expression.

    Returns
    -------
    :class:`.Expression`
    """
    wrapper = {'has_expr': False}
    def typecheck_expr(t, x):
        if isinstance(x, Expression):
            wrapper['has_expr'] = True
            if x.dtype != t:
                raise TypeError(f"'literal': type mismatch: expected '{t}', found '{x.dtype}'")
            elif x._indices.source is not None:
                if x._indices.axes:
                    raise ExpressionException(f"'literal' can only accept scalar or global expression arguments,"
                                              f" found indices {x._indices.axes}")
            return False
        elif x is None:
            return False
        else:
            t._typecheck_one_level(x)
            return True
    if dtype is None:
        dtype = impute_type(x)

    try:
        dtype._traverse(x, typecheck_expr)
    except TypeError as e:
        raise TypeError("'literal': object did not match the passed type '{}'"
                        .format(dtype)) from e

    if wrapper['has_expr']:
        return literal(hl.eval(to_expr(x, dtype)), dtype)

    if x is None:
        return hl.null(dtype)
    elif is_primitive(dtype):
        if dtype == tint32:
            assert isinstance(x, builtins.int)
            assert tint32.min_value <= x <= tint32.max_value
            return construct_expr(I32(x), tint32)
        elif dtype == tint64:
            assert isinstance(x, builtins.int)
            assert tint64.min_value <= x <= tint64.max_value
            return construct_expr(I64(x), tint64)
        elif dtype == tfloat32:
            assert isinstance(x, (builtins.float, builtins.int))
            return construct_expr(F32(x), tfloat32)
        elif dtype == tfloat64:
            assert isinstance(x, (builtins.float, builtins.int))
            return construct_expr(F64(x), tfloat64)
        elif dtype == tbool:
            assert isinstance(x, builtins.bool)
            return construct_expr(TrueIR() if x else FalseIR(), tbool)
        else:
            assert dtype == tstr
            assert isinstance(x, builtins.str)
            return construct_expr(Str(x), tstr)
    else:
        return construct_expr(Literal(dtype, x), dtype)

@typecheck(condition=expr_bool, consequent=expr_any, alternate=expr_any, missing_false=bool)
def cond(condition,
         consequent,
         alternate,
         missing_false: bool = False):
    """Expression for an if/else statement; tests a condition and returns one of two options based on the result.

    Examples
    --------

    >>> x = 5
    >>> hl.eval(hl.cond(x < 2, 'Hi', 'Bye'))
    'Bye'

    >>> a = hl.literal([1, 2, 3, 4])
    >>> hl.eval(hl.cond(hl.len(a) > 0, 2.0 * a, a / 2.0))
    [2.0, 4.0, 6.0, 8.0]

    Notes
    -----

    If `condition` evaluates to ``True``, returns `consequent`. If `condition`
    evaluates to ``False``, returns `alternate`. If `predicate` is missing, returns
    missing.

    Note
    ----
    The type of `consequent` and `alternate` must be the same.

    Parameters
    ----------
    condition : :class:`.BooleanExpression`
        Condition to test.
    consequent : :class:`.Expression`
        Branch to return if the condition is ``True``.
    alternate : :class:`.Expression`
        Branch to return if the condition is ``False``.
    missing_false : :obj:`.bool`
        If ``True``, treat missing `condition` as ``False``.

    See Also
    --------
    :func:`.case`, :func:`.switch`

    Returns
    -------
    :class:`.Expression`
        One of `consequent`, `alternate`, or missing, based on `condition`.
    """
    if missing_false:
        condition = hl.bind(lambda x: hl.is_defined(x) & x,
                            condition)
    indices, aggregations = unify_all(condition, consequent, alternate)

    consequent, alternate, success = unify_exprs(consequent, alternate)
    if not success:
        raise TypeError(f"'cond' requires the 'consequent' and 'alternate' arguments to have the same type\n"
                        f"    consequent: type '{consequent.dtype}'\n"
                        f"    alternate:  type '{alternate.dtype}'")
    assert consequent.dtype == alternate.dtype

    return construct_expr(If(condition._ir, consequent._ir, alternate._ir),
                              consequent.dtype, indices, aggregations)


def case(missing_false: bool=False) -> 'hail.expr.builders.CaseBuilder':
    """Chain multiple if-else statements with a :class:`.CaseBuilder`.

    Examples
    --------

    >>> x = hl.literal('foo bar baz')
    >>> expr = (hl.case()
    ...                  .when(x[:3] == 'FOO', 1)
    ...                  .when(hl.len(x) == 11, 2)
    ...                  .when(x == 'secret phrase', 3)
    ...                  .default(0))
    >>> hl.eval(expr)
    2

    Parameters
    ----------
    missing_false : :obj:`bool`
        Treat missing predicates as ``False``.

    See Also
    --------
    :class:`.CaseBuilder`, :func:`.switch`, :func:`.cond`

    Returns
    -------
    :class:`.CaseBuilder`.
    """
    from .builders import CaseBuilder
    return CaseBuilder(missing_false=missing_false)


@typecheck(expr=expr_any)
def switch(expr) -> 'hail.expr.builders.SwitchBuilder':
    """Build a conditional tree on the value of an expression.

    Examples
    --------

    >>> csq = hl.literal('loss of function')
    >>> expr = (hl.switch(csq)
    ...                  .when('synonymous', 1)
    ...                  .when('SYN', 1)
    ...                  .when('missense', 2)
    ...                  .when('MIS', 2)
    ...                  .when('loss of function', 3)
    ...                  .when('LOF', 3)
    ...                  .or_missing())
    >>> hl.eval(expr)
    3

    See Also
    --------
    :class:`.SwitchBuilder`, :func:`.case`, :func:`.cond`

    Parameters
    ----------
    expr : :class:`.Expression`
        Value to match against.

    Returns
    -------
    :class:`.SwitchBuilder`
    """
    from .builders import SwitchBuilder
    return SwitchBuilder(expr)


@typecheck(f=anytype, exprs=expr_any)
def bind(f: Callable, *exprs):
    """Bind a temporary variable and use it in a function.

    Examples
    --------

    >>> hl.eval(hl.bind(lambda x: x + 1, 1))
    2

    :func:`.bind` also can take multiple arguments:

    >>> hl.eval(hl.bind(lambda x, y: x / y, x, x))
    1.0

    Parameters
    ----------
    f : function ( (args) -> :class:`.Expression`)
        Function of `exprs`.
    exprs : variable-length args of :class:`.Expression`
        Expressions to bind.

    Returns
    -------
    :class:`.Expression`
        Result of evaluating `f` with `exprs` as arguments.
    """
    args = []
    uids = []
    irs = []

    for expr in exprs:
        uid = Env.get_uid()
        args.append(construct_variable(uid, expr._type, expr._indices, expr._aggregations))
        uids.append(uid)
        irs.append(expr._ir)

    lambda_result = to_expr(f(*args))
    indices, aggregations = unify_all(*exprs, lambda_result)

    res_ir = lambda_result._ir
    for (uid, ir) in builtins.zip(uids, irs):
        res_ir = Let(uid, ir, res_ir)

    return construct_expr(res_ir, lambda_result.dtype, indices, aggregations)


def rbind(*exprs):
    """Bind a temporary variable and use it in a function.

    This is :func:`.bind` with flipped argument order.

    Examples
    --------

    >>> hl.eval(hl.rbind(1, lambda x: x + 1))
    2

    :func:`.let` also can take multiple arguments:

    >>> hl.eval(hl.rbind(4.0, 2.0, lambda x, y: x / y))
    2.0

    Parameters
    ----------
    exprs : variable-length args of :class:`.Expression`
        Expressions to bind.
    f : function ( (args) -> :class:`.Expression`)
        Function of `exprs`.

    Returns
    -------
    :class:`.Expression`
        Result of evaluating `f` with `exprs` as arguments.
    """

    f = exprs[-1]
    args = [expr_any.check(arg, 'rbind', f'argument {index}') for index, arg in enumerate(exprs[:-1])]

    return hl.bind(f, *args)


@typecheck(c1=expr_int32, c2=expr_int32, c3=expr_int32, c4=expr_int32)
def chi_squared_test(c1, c2, c3, c4) -> StructExpression:
    """Performs chi-squared test of independence on a 2x2 contingency table.

    Examples
    --------

    >>> hl.eval(hl.chi_squared_test(10, 10, 10, 10))
    Struct(p_value=1.0, odds_ratio=1.0)

    >>> hl.eval(hl.chi_squared_test(51, 43, 22, 92))
    Struct(p_value=1.4626257805267089e-07, odds_ratio=4.959830866807611)

    Notes
    -----
    The odds ratio is given by ``(c1 / c2) / (c3 / c4)``.

    Returned fields may be ``nan`` or ``inf``.

    Parameters
    ----------
    c1 : int or :class:`.Expression` of type :py:data:`.tint32`
        Value for cell 1.
    c2 : int or :class:`.Expression` of type :py:data:`.tint32`
        Value for cell 2.
    c3 : int or :class:`.Expression` of type :py:data:`.tint32`
        Value for cell 3.
    c4 : int or :class:`.Expression` of type :py:data:`.tint32`
        Value for cell 4.

    Returns
    -------
    :class:`.StructExpression`
        A :class:`.tstruct` expression with two fields, `p_value`
        (:py:data:`.tfloat64`) and `odds_ratio` (:py:data:`.tfloat64`).
    """
    ret_type = tstruct(p_value=tfloat64, odds_ratio=tfloat64)
    return _func("chi_squared_test", ret_type, c1, c2, c3, c4)


@typecheck(c1=expr_int32, c2=expr_int32, c3=expr_int32, c4=expr_int32, min_cell_count=expr_int32)
def contingency_table_test(c1, c2, c3, c4, min_cell_count) -> StructExpression:
    """Performs chi-squared or Fisher's exact test of independence on a 2x2
    contingency table.

    Examples
    --------

    >>> hl.eval(hl.contingency_table_test(51, 43, 22, 92, min_cell_count=22))
    Struct(p_value=1.4626257805267089e-07, odds_ratio=4.959830866807611)

    >>> hl.eval(hl.contingency_table_test(51, 43, 22, 92, min_cell_count=23))
    Struct(p_value=2.1564999740157304e-07, odds_ratio=4.918058171469967)

    Notes
    -----
    If all cell counts are at least `min_cell_count`, the chi-squared test is
    used. Otherwise, Fisher's exact test is used.

    Returned fields may be ``nan`` or ``inf``.

    Parameters
    ----------
    c1 : int or :class:`.Expression` of type :py:data:`.tint32`
        Value for cell 1.
    c2 : int or :class:`.Expression` of type :py:data:`.tint32`
        Value for cell 2.
    c3 : int or :class:`.Expression` of type :py:data:`.tint32`
        Value for cell 3.
    c4 : int or :class:`.Expression` of type :py:data:`.tint32`
        Value for cell 4.
    min_cell_count : int or :class:`.Expression` of type :py:data:`.tint32`
        Minimum count in every cell to use the chi-squared test.

    Returns
    -------
    :class:`.StructExpression`
        A :class:`.tstruct` expression with two fields, `p_value`
        (:py:data:`.tfloat64`) and `odds_ratio` (:py:data:`.tfloat64`).
    """
    ret_type = tstruct(p_value=tfloat64, odds_ratio=tfloat64)
    return _func("contingency_table_test", ret_type, c1, c2, c3, c4, min_cell_count)


@typecheck(collection=expr_oneof(expr_dict(),
                                 expr_set(expr_tuple([expr_any, expr_any])),
                                 expr_array(expr_tuple([expr_any, expr_any]))))
def dict(collection) -> DictExpression:
    """Creates a dictionary.

    Examples
    --------

    >>> hl.eval(hl.dict([('foo', 1), ('bar', 2), ('baz', 3)]))  # doctest: +NOTEST
    {u'bar': 2, u'baz': 3, u'foo': 1}

    Notes
    -----
    This method expects arrays or sets with elements of type :class:`.ttuple`
    with 2 fields. The first field of the tuple becomes the key, and the second
    field becomes the value.

    Parameters
    ----------
    collection : :class:`.DictExpression` or :class:`.ArrayExpression` or :class:`.SetExpression`

    Returns
    -------
    :class:`.DictExpression`
    """
    if isinstance(collection.dtype, tarray) or isinstance(collection.dtype, tset):
        key_type, value_type = collection.dtype.element_type.types
        return _func('dict', tdict(key_type, value_type), collection)
    else:
        assert isinstance(collection.dtype, tdict)
        return collection


@typecheck(x=expr_float64, a=expr_float64, b=expr_float64)
def dbeta(x, a, b) -> Float64Expression:
    """
    Returns the probability density at `x` of a `beta distribution
    <https://en.wikipedia.org/wiki/Beta_distribution>`__ with parameters `a`
    (alpha) and `b` (beta).

    Examples
    --------

    >>> hl.eval(hl.dbeta(.2, 5, 20))
    4.900377563180943

    Parameters
    ----------
    x : :obj:`float` or :class:`.Expression` of type :py:data:`.tfloat64`
        Point in [0,1] at which to sample. If a < 1 then x must be positive.
        If b < 1 then x must be less than 1.
    a : :obj:`float` or :class:`.Expression` of type :py:data:`.tfloat64`
        The alpha parameter in the beta distribution. The result is undefined
        for non-positive a.
    b : :obj:`float` or :class:`.Expression` of type :py:data:`.tfloat64`
        The beta parameter in the beta distribution. The result is undefined
        for non-positive b.

    Returns
    -------
    :class:`.Float64Expression`
    """
    return _func("dbeta", tfloat64, x, a, b)


@typecheck(x=expr_float64, lamb=expr_float64, log_p=expr_bool)
def dpois(x, lamb, log_p=False) -> Float64Expression:
    """Compute the (log) probability density at x of a Poisson distribution with rate parameter `lamb`.

    Examples
    --------

    >>> hl.eval(hl.dpois(5, 3))
    0.10081881344492458

    Parameters
    ----------
    x : :obj:`float` or :class:`.Expression` of type :py:data:`.tfloat64`
        Non-negative number at which to compute the probability density.
    lamb : :obj:`float` or :class:`.Expression` of type :py:data:`.tfloat64`
        Poisson rate parameter. Must be non-negative.
    log_p : :obj:`bool` or :class:`.BooleanExpression`
        If true, the natural logarithm of the probability density is returned.

    Returns
    -------
    :class:`.Expression` of type :py:data:`.tfloat64`
        The (log) probability density.
    """
    return _func("dpois", tfloat64, x, lamb, log_p)


@typecheck(x=expr_float64)
def exp(x) -> Float64Expression:
    """Computes `e` raised to the power `x`.

    Examples
    --------

    >>> hl.eval(hl.exp(2))
    7.38905609893065

    Parameters
    ----------
    x : float or :class:`.Expression` of type :py:data:`.tfloat64`

    Returns
    -------
    :class:`.Expression` of type :py:data:`.tfloat64`
    """
    return _func("exp", tfloat64, x)


@typecheck(c1=expr_int32, c2=expr_int32, c3=expr_int32, c4=expr_int32)
def fisher_exact_test(c1, c2, c3, c4) -> StructExpression:
    """Calculates the p-value, odds ratio, and 95% confidence interval using
    Fisher's exact test for a 2x2 table.

    Examples
    --------

    >>> hl.eval(hl.fisher_exact_test(10, 10, 10, 10))
    Struct(p_value=1.0000000000000002, odds_ratio=1.0,
           ci_95_lower=0.24385796914260355, ci_95_upper=4.100747675033819)

    >>> hl.eval(hl.fisher_exact_test(51, 43, 22, 92))
    Struct(p_value=2.1564999740157304e-07, odds_ratio=4.918058171469967,
           ci_95_lower=2.5659373368248444, ci_95_upper=9.677929632035475)

    Notes
    -----
    This method is identical to the version implemented in
    `R <https://stat.ethz.ch/R-manual/R-devel/library/stats/html/fisher.test.html>`_ with default
    parameters (two-sided, alpha = 0.05, null hypothesis that the odds ratio equals 1).

    Returned fields may be ``nan`` or ``inf``.

    Parameters
    ----------
    c1 : int or :class:`.Expression` of type :py:data:`.tint32`
        Value for cell 1.
    c2 : int or :class:`.Expression` of type :py:data:`.tint32`
        Value for cell 2.
    c3 : int or :class:`.Expression` of type :py:data:`.tint32`
        Value for cell 3.
    c4 : int or :class:`.Expression` of type :py:data:`.tint32`
        Value for cell 4.

    Returns
    -------
    :class:`.StructExpression`
        A :class:`.tstruct` expression with four fields, `p_value`
        (:py:data:`.tfloat64`), `odds_ratio` (:py:data:`.tfloat64`),
        `ci_95_lower (:py:data:`.tfloat64`), and `ci_95_upper`
        (:py:data:`.tfloat64`).
    """
    ret_type = tstruct(p_value=tfloat64,
                       odds_ratio=tfloat64,
                       ci_95_lower=tfloat64,
                       ci_95_upper=tfloat64)
    return _func("fisher_exact_test", ret_type, c1, c2, c3, c4)


@typecheck(x=expr_oneof(expr_float32, expr_float64))
def floor(x):
    """The largest integral value that is less than or equal to `x`.

    Examples
    --------

    >>> hl.eval(hl.floor(3.1))
    3.0

    Parameters
    ----------
    x : :class:`.Float32Expression` or :class:`.Float64Expression`

    Returns
    -------
    :class:`.Float32Expression` or :class:`.Float64Expression`
    """
    return _func("floor", x.dtype, x)


@typecheck(x=expr_oneof(expr_float32, expr_float64))
def ceil(x):
    """The smallest integral value that is greater than or equal to `x`.

    Examples
    --------

    >>> hl.eval(hl.ceil(3.1))
    4.0

    Parameters
    ----------
    x : :class:`.Float32Expression` or :class:`.Float64Expression`

    Returns
    -------
    :class:`.Float32Expression` or :class:`.Float64Expression`
    """
    return _func("ceil", x.dtype, x)


@typecheck(n_hom_ref=expr_int32, n_het=expr_int32, n_hom_var=expr_int32)
def hardy_weinberg_test(n_hom_ref, n_het, n_hom_var) -> StructExpression:
    """Performs test of Hardy-Weinberg equilibrium.

    Examples
    --------

    >>> hl.eval(hl.hardy_weinberg_test(250, 500, 250))
    Struct(het_freq_hwe=0.5002501250625313, p_value=0.9747844394217698)

    >>> hl.eval(hl.hardy_weinberg_test(37, 200, 85))
    Struct(het_freq_hwe=0.48964964307448583, p_value=1.1337210383168987e-06)

    Notes
    -----
    This method performs a two-sided exact test with mid-p-value correction of
    `Hardy-Weinberg equilibrium <https://en.wikipedia.org/wiki/Hardy%E2%80%93Weinberg_principle>`__
    via an efficient implementation of the
    `Levene-Haldane distribution <https://hail.is/docs/0.2/LeveneHaldane.pdf>`__,
    which models the number of heterozygous individuals under equilibrium.

    The mean of this distribution is ``(n_hom_ref * n_hom_var) / (2n - 1)`` where
    ``n = n_hom_ref + n_het + n_hom_var``. So the expected frequency of heterozygotes
    under equilibrium, `het_freq_hwe`, is this mean divided by ``n``.

    Parameters
    ----------
    n_hom_ref : int or :class:`.Expression` of type :py:data:`.tint32`
        Number of homozygous reference genotypes.
    n_het : int or :class:`.Expression` of type :py:data:`.tint32`
        Number of heterozygous genotypes.
    n_hom_var : int or :class:`.Expression` of type :py:data:`.tint32`
        Number of homozygous variant genotypes.

    Returns
    -------
    :class:`.StructExpression`
        A struct expression with two fields, `het_freq_hwe`
        (:py:data:`.tfloat64`) and `p_value` (:py:data:`.tfloat64`).
    """
    ret_type = tstruct(het_freq_hwe=tfloat64,
                       p_value=tfloat64)
    return _func("hardy_weinberg_test", ret_type, n_hom_ref, n_het, n_hom_var)


@typecheck(contig=expr_str, pos=expr_int32,
           reference_genome=reference_genome_type)
def locus(contig, pos, reference_genome: Union[str, ReferenceGenome] = 'default') -> LocusExpression:
    """Construct a locus expression from a chromosome and position.

    Examples
    --------

    >>> hl.eval(hl.locus("1", 10000, reference_genome='GRCh37'))
    Locus(contig=1, position=10000, reference_genome=GRCh37)

    Parameters
    ----------
    contig : str or :class:`.StringExpression`
        Chromosome.
    pos : int or :class:`.Expression` of type :py:data:`.tint32`
        Base position along the chromosome.
    reference_genome : :obj:`str` or :class:`.ReferenceGenome`
        Reference genome to use.

    Returns
    -------
    :class:`.LocusExpression`
    """
    fname = 'Locus({})'.format(reference_genome.name)
    return _func(fname, tlocus(reference_genome), contig, pos)


@typecheck(global_pos=expr_int64,
           reference_genome=reference_genome_type)
def locus_from_global_position(global_pos,
                               reference_genome: Union[str, ReferenceGenome] = 'default') -> LocusExpression:
    """Constructs a locus expression from a global position and a reference genome.
    The inverse of :meth:`.LocusExpression.global_position`.

    Examples
    --------
    >>> hl.eval(hl.locus_from_global_position(0))
    Locus(contig=1, position=1, reference_genome=GRCh37)

    >>> hl.eval(hl.locus_from_global_position(2824183054))
    Locus(contig=21, position=42584230, reference_genome=GRCh37)

    >>> hl.eval(hl.locus_from_global_position(2824183054, reference_genome='GRCh38'))
    Locus(contig=chr22, position=1, reference_genome=GRCh38)

    Parameters
    ----------
    global_pos : int or :class:`.Expression` of type :py:data:`.tint64`
        Global base position along the reference genome.
    reference_genome : :obj:`str` or :class:`.ReferenceGenome`
        Reference genome to use for converting the global position to a contig and local position.

    Returns
    -------
    :class:`.LocusExpression`
    """
    fname = 'globalPosToLocus({})'.format(reference_genome.name)
    return _func(fname, tlocus(reference_genome), global_pos)


@typecheck(s=expr_str,
           reference_genome=reference_genome_type)
def parse_locus(s, reference_genome: Union[str, ReferenceGenome] = 'default') -> LocusExpression:
    """Construct a locus expression by parsing a string or string expression.

    Examples
    --------

    >>> hl.eval(hl.parse_locus('1:10000', reference_genome='GRCh37'))
    Locus(contig=1, position=10000, reference_genome=GRCh37)

    Notes
    -----
    This method expects strings of the form ``contig:position``, e.g. ``16:29500000``
    or ``X:123456``.

    Parameters
    ----------
    s : str or :class:`.StringExpression`
        String to parse.
    reference_genome : :obj:`str` or :class:`.ReferenceGenome`
        Reference genome to use.

    Returns
    -------
    :class:`.LocusExpression`
    """
    return _func('Locus({})'.format(reference_genome.name), tlocus(reference_genome), s)


@typecheck(s=expr_str,
           reference_genome=reference_genome_type)
def parse_variant(s, reference_genome: Union[str, ReferenceGenome] = 'default') -> StructExpression:
    """Construct a struct with a locus and alleles by parsing a string.

    Examples
    --------

    >>> hl.eval(hl.parse_variant('1:100000:A:T,C', reference_genome='GRCh37'))
    Struct(locus=Locus(contig=1, position=100000, reference_genome=GRCh37), alleles=['A', 'T', 'C'])

    Notes
    -----
    This method returns an expression of type :class:`.tstruct` with the
    following fields:

     - `locus` (:class:`.tlocus`)
     - `alleles` (:class:`.tarray` of :py:data:`.tstr`)

    Parameters
    ----------
    s : :class:`.StringExpression`
        String to parse.
    reference_genome: :obj:`str` or :class:`.ReferenceGenome`
        Reference genome to use.

    Returns
    -------
    :class:`.StructExpression`
        Struct with fields `locus` and `alleles`.
    """
    t = tstruct(locus=tlocus(reference_genome),
                alleles=tarray(tstr))
    return _func('LocusAlleles({})'.format(reference_genome.name), t, s)


def variant_str(*args) -> 'StringExpression':
    """Create a variant colon-delimited string.

    Parameters
    ----------
    args
        Arguments (see notes).

    Returns
    -------
    :class:`.StringExpression`

    Notes
    -----
    Expects either one argument of type
    ``struct{locus: locus<RG>, alleles: array<str>``, or two arguments of type
    ``locus<RG>`` and ``array<str>``. The function returns a string of the form

    .. code-block:: text

        CHR:POS:REF:ALT1,ALT2,...ALTN
        e.g.
        1:1:A:T
        16:250125:AAA:A,CAA

    Examples
    --------
    >>> hl.eval(hl.variant_str(hl.locus('1', 10000), ['A', 'T', 'C']))
    '1:10000:A:T,C'
    """
    args = [to_expr(arg) for arg in args]

    def type_error():
        raise ValueError(f"'variant_str' expects arguments of the following types:\n"
                         f"  Option 1: 1 argument of type 'struct{{locus: locus<RG>, alleles: array<str>}}\n"
                         f"  Option 2: 2 arguments of type 'locus<RG>', 'array<str>'\n"
                         f"  Found: {builtins.len(args)} {plural('argument', builtins.len(args))} "
                         f"of type {', '.join(builtins.str(x.dtype) for x in args)}")

    if builtins.len(args) == 1:
        [s] = args
        t = s.dtype
        if not isinstance(t, tstruct) \
                or not builtins.len(t) == 2 \
                or not isinstance(t[0], tlocus) \
                or not t[1] == tarray(tstr):
            type_error()
        return hl.rbind(s, lambda x: hl.str(x[0]) + ":" + x[1][0] + ":" + hl.delimit(x[1][1:]))
    elif builtins.len(args) == 2:
        [locus, alleles] = args
        if not isinstance(locus.dtype, tlocus) or not alleles.dtype == tarray(tstr):
            type_error()
        return hl.str(locus) + ":" + hl.rbind(alleles, lambda x: x[0] + ":" + hl.delimit(x[1:]))
    else:
        type_error()


@typecheck(gp=expr_array(expr_float64))
def gp_dosage(gp) -> Float64Expression:
    """
    Return expected genotype dosage from array of genotype probabilities.

    Examples
    --------

    >>> hl.eval(hl.gp_dosage([0.0, 0.5, 0.5]))
    1.5

    Notes
    -----
    This function is only defined for bi-allelic variants. The `gp` argument
    must be length 3. The value is ``gp[1] + 2 * gp[2]``.

    Parameters
    ----------
    gp : :class:`.ArrayFloat64Expression`
        Length 3 array of bi-allelic genotype probabilities

    Returns
    -------
    :class:`.Expression` of type :py:data:`.tfloat64`
    """
    return _func("dosage", tfloat64, gp)


@typecheck(pl=expr_array(expr_int32))
def pl_dosage(pl) -> Float64Expression:
    r"""
    Return expected genotype dosage from array of Phred-scaled genotype
    likelihoods with uniform prior. Only defined for bi-allelic variants. The
    `pl` argument must be length 3.

    For a PL array ``[a, b, c]``, let:

    .. math::

        a^\prime = 10^{-a/10} \\
        b^\prime = 10^{-b/10} \\
        c^\prime = 10^{-c/10} \\

    The genotype dosage is given by:

    .. math::

        \frac{b^\prime + 2 c^\prime}
             {a^\prime + b^\prime +c ^\prime}

    Examples
    --------

    >>> hl.eval(hl.pl_dosage([5, 10, 100]))
    0.24025307377482674

    Parameters
    ----------
    pl : :class:`.ArrayNumericExpression` of type :py:data:`.tint32`
        Length 3 array of bi-allelic Phred-scaled genotype likelihoods

    Returns
    -------
    :class:`.Expression` of type :py:data:`.tfloat64`
    """
    return hl.sum(pl_to_gp(pl) * [0, 1, 2], filter_missing=False)


@typecheck(pl=expr_array(expr_int32), _cache_size=int)
def pl_to_gp(pl, _cache_size=2048) -> ArrayNumericExpression:
    """
    Return the linear-scaled genotype probabilities from an array of Phred-scaled genotype likelihoods.

    Examples
    --------
    >>> hl.eval(hl.pl_to_gp([0, 10, 100]))
    [0.9090909090082644, 0.09090909090082644, 9.090909090082645e-11]

    Notes
    -----
    This function assumes a uniform prior on the possible genotypes.

    Parameters
    ----------
    pl : :class:`.ArrayNumericExpression` of type :py:data:`.tint32`
        Array of Phred-scaled genotype likelihoods.

    Returns
    -------
   :class:`.ArrayNumericExpression` of type :py:data:`.tfloat64`
    """
    phred_table = hl.literal([10 ** (-x/10.0) for x in builtins.range(_cache_size)])
    gp = hl.bind(lambda pls: pls.map(lambda x: hl.cond(x >= _cache_size, 10 ** (-x/10.0), phred_table[x])), pl)
    return hl.bind(lambda gp: gp / hl.sum(gp), gp)


@typecheck(start=expr_any, end=expr_any,
           includes_start=expr_bool, includes_end=expr_bool)
def interval(start,
             end,
             includes_start=True,
             includes_end=False) -> IntervalExpression:
    """Construct an interval expression.

    Examples
    --------

    >>> hl.eval(hl.interval(5, 100))
    Interval(start=5, end=100, includes_start=True, includes_end=False)

    >>> hl.eval(hl.interval(hl.locus("1", 100), hl.locus("1", 1000)))
        Interval(start=Locus(contig=1, position=100, reference_genome=GRCh37),
                 end=Locus(contig=1, position=1000, reference_genome=GRCh37),
                 includes_start=True,
                 includes_end=False)

    Notes
    -----
    `start` and `end` must have the same type.

    Parameters
    ----------
    start : :class:`.Expression`
        Start point.
    end : :class:`.Expression`
        End point.
    includes_start : :class:`.BooleanExpression`
        If ``True``, interval includes start point.
    includes_end : :class:`.BooleanExpression`
        If ``True``, interval includes end point.

    Returns
    -------
    :class:`.IntervalExpression`
    """
    if not start.dtype == end.dtype:
        raise TypeError("Type mismatch of start and end points: '{}', '{}'".format(start.dtype, end.dtype))

    return _func('Interval', tinterval(start.dtype), start, end, includes_start, includes_end)

@typecheck(contig=expr_str, start=expr_int32,
           end=expr_int32, includes_start=expr_bool,
           includes_end=expr_bool, reference_genome=reference_genome_type)
def locus_interval(contig,
                   start,
                   end,
                   includes_start=True,
                   includes_end=False,
                   reference_genome: Union[str, ReferenceGenome] = 'default') -> IntervalExpression:
    """Construct a locus interval expression.

    Examples
    --------

    >>> hl.eval(hl.locus_interval("1", 100, 1000, reference_genome='GRCh37'))
    Interval(start=Locus(contig=1, position=100, reference_genome=GRCh37),
             end=Locus(contig=1, position=1000, reference_genome=GRCh37),
             includes_start=True,
             includes_end=False)

    Parameters
    ----------
    contig : :class:`.StringExpression`
        Contig name.
    start : :class:`.Int32Expression`
        Starting base position.
    end : :class:`.Int32Expression`
        End base position.
    includes_start : :class:`.BooleanExpression`
        If ``True``, interval includes start point.
    includes_end : :class:`.BooleanExpression`
        If ``True``, interval includes end point.
    reference_genome : :obj:`str` or :class:`.hail.genetics.ReferenceGenome`
        Reference genome to use.

    Returns
    -------
    :class:`.IntervalExpression`
    """
    fname = 'LocusInterval({})'.format(reference_genome.name)
    return _func(fname, tinterval(tlocus(reference_genome)), contig, start, end, includes_start, includes_end)


@typecheck(s=expr_str,
           reference_genome=reference_genome_type)
def parse_locus_interval(s, reference_genome: Union[str, ReferenceGenome] = 'default') -> IntervalExpression:
    """Construct a locus interval expression by parsing a string or string
    expression.

    Examples
    --------

    >>> hl.eval(hl.parse_locus_interval('1:1000-2000', reference_genome='GRCh37'))
    Interval(start=Locus(contig=1, position=1000, reference_genome=GRCh37),
             end=Locus(contig=1, position=2000, reference_genome=GRCh37),
             includes_start=True,
             includes_end=False)

    >>> hl.eval(hl.parse_locus_interval('1:start-10M', reference_genome='GRCh37'))
    Interval(start=Locus(contig=1, position=1, reference_genome=GRCh37),
             end=Locus(contig=1, position=10000000, reference_genome=GRCh37),
             includes_start=True,
             includes_end=False)

    Notes
    -----
    The start locus must precede the end locus. The default bounds of the
    interval are left-inclusive and right-exclusive. To change this, add
    one of ``[`` or ``(`` at the beginning of the string for left-inclusive
    or left-exclusive respectively. Likewise, add one of ``]`` or ``)`` at
    the end of the string for right-inclusive or right-exclusive
    respectively.

    There are several acceptable representations for `s`.

    ``CHR1:POS1-CHR2:POS2`` is the fully specified representation, and
    we use this to define the various shortcut representations.

    In a ``POS`` field, ``start`` (``Start``, ``START``) stands for 1.

    In a ``POS`` field, ``end`` (``End``, ``END``) stands for the contig length.

    In a ``POS`` field, the qualifiers ``m`` (``M``) and ``k`` (``K``) multiply
    the given number by ``1,000,000`` and ``1,000``, respectively.  ``1.6K`` is
    short for 1600, and ``29M`` is short for 29000000.

    ``CHR:POS1-POS2`` stands for ``CHR:POS1-CHR:POS2``

    ``CHR1-CHR2`` stands for ``CHR1:START-CHR2:END``

    ``CHR`` stands for ``CHR:START-CHR:END``

    Note
    ----
        The bounds of the interval must be valid loci for the reference genome
        (contig in reference genome and position is within the range [1-END])
        except in the case where the position is ``0`` **AND** the interval is
        **left-exclusive** which is normalized to be ``1`` and left-inclusive.
        Likewise, in the case where the position is ``END + 1`` **AND**
        the interval is **right-exclusive** which is normalized to be ``END``
        and right-inclusive.

    Parameters
    ----------
    s : str or :class:`.StringExpression`
        String to parse.
    reference_genome : :obj:`str` or :class:`.hail.genetics.ReferenceGenome`
        Reference genome to use.

    Returns
    -------
    :class:`.IntervalExpression`
    """
    return _func('LocusInterval({})'.format(reference_genome.name),
                 tinterval(tlocus(reference_genome)), s)


@typecheck(alleles=expr_int32,
           phased=expr_bool)
def call(*alleles, phased=False) -> CallExpression:
    """Construct a call expression.

    Examples
    --------

    >>> hl.eval(hl.call(1, 0))
    Call(alleles=[0, 1], phased=False)

    Parameters
    ----------
    alleles : variable-length args of :obj:`int` or :class:`.Expression` of type :py:data:`.tint32`
        List of allele indices.
    phased : :obj:`bool`
        If ``True``, preserve the order of `alleles`.

    Returns
    -------
    :class:`.CallExpression`
    """
    if builtins.len(alleles) > 2:
        raise NotImplementedError("'call' supports a maximum of 2 alleles.")
    return _func('Call', tcall, *alleles, phased)


@typecheck(gt_index=expr_int32)
def unphased_diploid_gt_index_call(gt_index) -> CallExpression:
    """Construct an unphased, diploid call from a genotype index.

    Examples
    --------

    >>> hl.eval(hl.unphased_diploid_gt_index_call(4))
    Call(alleles=[1, 2], phased=False)

    Parameters
    ----------
    gt_index : :obj:`int` or :class:`.Expression` of type :py:data:`.tint32`
        Unphased, diploid genotype index.

    Returns
    -------
    :class:`.CallExpression`
    """
    return _func('UnphasedDiploidGtIndexCall', tcall, to_expr(gt_index))


@typecheck(s=expr_str)
def parse_call(s) -> CallExpression:
    """Construct a call expression by parsing a string or string expression.

    Examples
    --------

    >>> hl.eval(hl.parse_call('0|2'))
    Call(alleles=[0, 2], phased=True)

    Notes
    -----
    This method expects strings in the following format:

    +--------+-----------------+-----------------+
    | ploidy | Phased          | Unphased        |
    +========+=================+=================+
    |   0    | ``|-``          | ``-``           |
    +--------+-----------------+-----------------+
    |   1    | ``|i``          | ``i``           |
    +--------+-----------------+-----------------+
    |   2    | ``i|j``         | ``i/j``         |
    +--------+-----------------+-----------------+
    |   3    | ``i|j|k``       | ``i/j/k``       |
    +--------+-----------------+-----------------+
    |   N    | ``i|j|k|...|N`` | ``i/j/k/.../N`` |
    +--------+-----------------+-----------------+

    Parameters
    ----------
    s : str or :class:`.StringExpression`
        String to parse.

    Returns
    -------
    :class:`.CallExpression`
    """
    return _func('Call', tcall, s)


@typecheck(expression=expr_any)
def is_defined(expression) -> BooleanExpression:
    """Returns ``True`` if the argument is not missing.

    Examples
    --------

    >>> hl.eval(hl.is_defined(5))
    True

    >>> hl.eval(hl.is_defined(hl.null(hl.tstr)))
    False

    >>> hl.eval(hl.is_defined(hl.null(hl.tbool) & True))
    False

    Parameters
    ----------
    expression
        Expression to test.

    Returns
    -------
    :class:`.BooleanExpression`
        ``True`` if `expression` is not missing, ``False`` otherwise.
    """
    return ~apply_expr(lambda x: IsNA(x), tbool, expression)


@typecheck(expression=expr_any)
def is_missing(expression) -> BooleanExpression:
    """Returns ``True`` if the argument is missing.

    Examples
    --------

    >>> hl.eval(hl.is_missing(5))
    False

    >>> hl.eval(hl.is_missing(hl.null(hl.tstr)))
    True

    >>> hl.eval(hl.is_missing(hl.null(hl.tbool) & True))
    True

    Parameters
    ----------
    expression
        Expression to test.

    Returns
    -------
    :class:`.BooleanExpression`
        ``True`` if `expression` is missing, ``False`` otherwise.
    """
    return apply_expr(lambda x: IsNA(x), tbool, expression)


@typecheck(x=expr_oneof(expr_float32, expr_float64))
def is_nan(x) -> BooleanExpression:
    """Returns ``True`` if the argument is ``nan`` (not a number).

    Examples
    --------

    >>> hl.eval(hl.is_nan(0))
    False

    >>> hl.eval(hl.is_nan(hl.literal(0) / 0))
    True

    >>> hl.eval(hl.is_nan(hl.literal(0) / hl.null(hl.tfloat64)))
    None

    Notes
    -----
    Note that :meth:`.is_missing` will return ``False`` on ``nan`` since ``nan``
    is a defined value. Additionally, this method will return missing if `x` is
    missing.

    Parameters
    ----------
    x : float or :class:`.Expression` of type :py:data:`.tfloat64`
        Expression to test.

    Returns
    -------
    :class:`.BooleanExpression`
        ``True`` if `x` is ``nan``, ``False`` otherwise.
    """
    return _func("isnan", tbool, x)

@typecheck(x=expr_oneof(expr_float32, expr_float64))
def is_finite(x) -> BooleanExpression:
    """Returns ``True`` if the argument is a finite floating-point number.

    Examples
    --------
    >>> hl.eval(hl.is_finite(0))
    True

    >>> hl.eval(hl.is_finite(float('nan')))
    False

    >>> hl.eval(hl.is_finite(float('inf')))
    False

    >>> hl.eval(hl.is_finite(hl.null('float32')))
    None

    Notes
    -----
    This method will return missing, not ``True``, if `x` is missing.

    Parameters
    ----------
    x : float or :class:`.Expression` of type :py:data:`.tfloat64`

    Returns
    -------
    :class:`.BooleanExpression`
    """
    return _func("is_finite", tbool, x)

@typecheck(x=expr_oneof(expr_float32, expr_float64))
def is_infinite(x) -> BooleanExpression:
    """Returns ``True`` if the argument is positive or negative infinity.

    Examples
    --------
    >>> hl.eval(hl.is_infinite(0))
    False

    >>> hl.eval(hl.is_infinite(float('nan')))
    False

    >>> hl.eval(hl.is_infinite(float('inf')))
    True

    >>> hl.eval(hl.is_infinite(hl.null('float32')))
    None

    Notes
    -----
    This method will return missing, not ``False``, if `x` is missing.

    Parameters
    ----------
    x : float or :class:`.Expression` of type :py:data:`.tfloat64`

    Returns
    -------
    :class:`.BooleanExpression`
    """
    return _func("is_infinite", tbool, x)


@typecheck(x=expr_any)
def json(x) -> StringExpression:
    """Convert an expression to a JSON string expression.

    Examples
    --------

    >>> hl.eval(hl.json([1,2,3,4,5]))
    '[1,2,3,4,5]'

    >>> hl.eval(hl.json(hl.struct(a='Hello', b=0.12345, c=[1,2], d={'hi', 'bye'})))  # doctest: +NOTEST
    '{"a":"Hello","c":[1,2],"b":0.12345,"d":["bye","hi"]}'

    Parameters
    ----------
    x
        Expression to convert.

    Returns
    -------
    :class:`.StringExpression`
        String expression with JSON representation of `x`.
    """
    return _func("json", tstr, x)


@typecheck(x=expr_float64, base=nullable(expr_float64))
def log(x, base=None) -> Float64Expression:
    """Take the logarithm of the `x` with base `base`.

    Examples
    --------

    >>> hl.eval(hl.log(10))
    2.302585092994046

    >>> hl.eval(hl.log(10, 10))
    1.0

    >>> hl.eval(hl.log(1024, 2))
    10.0

    Notes
    -----
    If the `base` argument is not supplied, then the natural logarithm is used.

    Parameters
    ----------
    x : float or :class:`.Expression` of type :py:data:`.tfloat64`
    base : float or :class:`.Expression` of type :py:data:`.tfloat64`

    Returns
    -------
    :class:`.Expression` of type :py:data:`.tfloat64`
    """
    x = to_expr(x)
    if base is not None:
        return _func("log", tfloat64, x, to_expr(base))
    else:
        return _func("log", tfloat64, x)


@typecheck(x=expr_float64)
def log10(x) -> Float64Expression:
    """Take the logarithm of the `x` with base 10.

    Examples
    --------

    >>> hl.eval(hl.log10(1000))
    3.0

    >>> hl.eval(hl.log10(0.0001123))
    -3.949620243738542

    Parameters
    ----------
    x : float or :class:`.Expression` of type :py:data:`.tfloat64`

    Returns
    -------
    :class:`.Expression` of type :py:data:`.tfloat64`
    """
    return _func("log10", tfloat64, x)


@typecheck(args=expr_any)
def coalesce(*args):
    """Returns the first non-missing value of `args`.

    Examples
    --------

    >>> x1 = hl.null('int')
    >>> x2 = 2
    >>> hl.eval(hl.coalesce(x1, x2))
    2

    Notes
    -----
    All arguments must have the same type, or must be convertible to a common
    type (all numeric, for instance).

    See Also
    --------
    :func:`.or_else`

    Parameters
    ----------
    args : variable-length args of :class:`.Expression`

    Returns
    -------
    :class:`.Expression`
    """
    if builtins.len(args) < 1:
        raise ValueError("'coalesce' requires at least one expression argument")
    *exprs, success = unify_exprs(*args)
    if not success:
        arg_types = ''.join([f"\n    argument {i}: type '{arg.dtype}'" for i, arg in enumerate(exprs)])
        raise TypeError(f"'coalesce' requires all arguments to have the same type or compatible types"
                        f"{arg_types}")
    return functools.reduce(lambda x, y: hl.or_else(y, x), exprs[::-1])

@typecheck(a=expr_any, b=expr_any)
def or_else(a, b):
    """If `a` is missing, return `b`.

    Examples
    --------

    >>> hl.eval(hl.or_else(5, 7))
    5

    >>> hl.eval(hl.or_else(hl.null(hl.tint32), 7))
    7

    See Also
    --------
    :func:`.coalesce`

    Parameters
    ----------
    a: :class:`.Expression`
    b: :class:`.Expression`

    Returns
    -------
    :class:`.Expression`
    """
    a, b, success = unify_exprs(a, b)
    if not success:
        raise TypeError(f"'or_else' requires the 'a' and 'b' arguments to have the same type\n"
                        f"    a: type '{a.dtype}'\n"
                        f"    b: type '{b.dtype}'")
    assert a.dtype == b.dtype
    return hl.rbind(a, lambda aa: hl.cond(hl.is_defined(aa), aa, b))

@typecheck(predicate=expr_bool, value=expr_any)
def or_missing(predicate, value):
    """Returns `value` if `predicate` is ``True``, otherwise returns missing.

    Examples
    --------

    >>> hl.eval(hl.or_missing(True, 5))
    5

    >>> hl.eval(hl.or_missing(False, 5))
    None

    Parameters
    ----------
    predicate : :class:`.BooleanExpression`
    value : :class:`.Expression`
        Value to return if `predicate` is ``True``.

    Returns
    -------
    :class:`.Expression`
        This expression has the same type as `b`.
    """

    return hl.cond(predicate, value, hl.null(value.dtype))


@typecheck(x=expr_int32, n=expr_int32, p=expr_float64,
           alternative=enumeration("two.sided", "greater", "less"))
def binom_test(x, n, p, alternative: str) -> Float64Expression:
    """Performs a binomial test on `p` given `x` successes in `n` trials.

    Examples
    --------

    >>> hl.eval(hl.binom_test(5, 10, 0.5, 'less'))
    0.6230468749999999

    With alternative ``less``, the p-value is the probability of at most `x`
    successes, i.e. the cumulative probability at `x` of the distribution
    Binom(`n`, `p`). With ``greater``, the p-value is the probably of at least
    `x` successes. With ``two.sided``, the p-value is the total probability of
    all outcomes with probability at most that of `x`.

    Returns the p-value from the `exact binomial test
    <https://en.wikipedia.org/wiki/Binomial_test>`__ of the null hypothesis that
    success has probability `p`, given `x` successes in `n` trials.

    Parameters
    ----------
    x : int or :class:`.Expression` of type :py:data:`.tint32`
        Number of successes.
    n : int or :class:`.Expression` of type :py:data:`.tint32`
        Number of trials.
    p : float or :class:`.Expression` of type :py:data:`.tfloat64`
        Probability of success, between 0 and 1.
    alternative
        : One of, "two.sided", "greater", "less".

    Returns
    -------
    :class:`.Expression` of type :py:data:`.tfloat64`
        p-value.
    """

    alt_enum = {"two.sided": 0, "greater": 1, "less": 2}[alternative]
    return _func("binomTest", tfloat64, x, n, p, to_expr(alt_enum))


@typecheck(x=expr_float64, df=expr_float64)
def pchisqtail(x, df) -> Float64Expression:
    """Returns the probability under the right-tail starting at x for a chi-squared
    distribution with df degrees of freedom.

    Examples
    --------

    >>> hl.eval(hl.pchisqtail(5, 1))
    0.025347318677468304

    Parameters
    ----------
    x : float or :class:`.Expression` of type :py:data:`.tfloat64`
    df : float or :class:`.Expression` of type :py:data:`.tfloat64`
        Degrees of freedom.

    Returns
    -------
    :class:`.Expression` of type :py:data:`.tfloat64`
    """
    return _func("pchisqtail", tfloat64, x, df)


@typecheck(x=expr_float64)
def pnorm(x) -> Float64Expression:
    """The cumulative probability function of a standard normal distribution.

    Examples
    --------

    >>> hl.eval(hl.pnorm(0))
    0.5

    >>> hl.eval(hl.pnorm(1))
    0.8413447460685429

    >>> hl.eval(hl.pnorm(2))
    0.9772498680518208

    Notes
    -----
    Returns the left-tail probability `p` = Prob(:math:Z < x) with :math:Z a standard normal random variable.

    Parameters
    ----------
    x : float or :class:`.Expression` of type :py:data:`.tfloat64`

    Returns
    -------
    :class:`.Expression` of type :py:data:`.tfloat64`
    """
    return _func("pnorm", tfloat64, x)


@typecheck(x=expr_float64, lamb=expr_float64, lower_tail=expr_bool, log_p=expr_bool)
def ppois(x, lamb, lower_tail=True, log_p=False) -> Float64Expression:
    r"""The cumulative probability function of a Poisson distribution.

    Examples
    --------

    >>> hl.eval(hl.ppois(2, 1))
    0.9196986029286058

    Notes
    -----
    If `lower_tail` is true, returns Prob(:math:`X \leq` `x`) where :math:`X` is a
    Poisson random variable with rate parameter `lamb`. If `lower_tail` is false,
    returns Prob(:math:`X` > `x`).

    Parameters
    ----------
    x : float or :class:`.Expression` of type :py:data:`.tfloat64`
    lamb : float or :class:`.Expression` of type :py:data:`.tfloat64`
        Rate parameter of Poisson distribution.
    lower_tail : bool or :class:`.BooleanExpression`
        If ``True``, compute the probability of an outcome at or below `x`,
        otherwise greater than `x`.
    log_p : bool or :class:`.BooleanExpression`
        Return the natural logarithm of the probability.

    Returns
    -------
    :class:`.Expression` of type :py:data:`.tfloat64`
    """
    return _func("ppois", tfloat64, x, lamb, lower_tail, log_p)


@typecheck(p=expr_float64, df=expr_float64)
def qchisqtail(p, df) -> Float64Expression:
    """Inverts :meth:`.pchisqtail`.

    Examples
    --------

    >>> hl.eval(hl.qchisqtail(0.01, 1))
    6.634896601021213

    Notes
    -----
    Returns right-quantile `x` for which `p` = Prob(:math:`Z^2` > x) with :math:`Z^2` a chi-squared random
     variable with degrees of freedom specified by `df`. `p` must satisfy 0 < `p` <= 1.

    Parameters
    ----------
    p : float or :class:`.Expression` of type :py:data:`.tfloat64`
        Probability.
    df : float or :class:`.Expression` of type :py:data:`.tfloat64`
        Degrees of freedom.

    Returns
    -------
    :class:`.Expression` of type :py:data:`.tfloat64`
    """
    return _func("qchisqtail", tfloat64, p, df)


@typecheck(p=expr_float64)
def qnorm(p) -> Float64Expression:
    """Inverts :meth:`.pnorm`.

    Examples
    --------

    >>> hl.eval(hl.qnorm(0.90))
    1.2815515655446008

    Notes
    -----
    Returns left-quantile `x` for which p = Prob(:math:`Z` < x) with :math:`Z` a standard normal random variable.
    `p` must satisfy 0 < `p` < 1.

    Parameters
    ----------
    p : float or :class:`.Expression` of type :py:data:`.tfloat64`
        Probability.

    Returns
    -------
    :class:`.Expression` of type :py:data:`.tfloat64`
    """
    return _func("qnorm", tfloat64, p)


@typecheck(p=expr_float64, lamb=expr_float64, lower_tail=expr_bool, log_p=expr_bool)
def qpois(p, lamb, lower_tail=True, log_p=False) -> Float64Expression:
    r"""Inverts :meth:`.ppois`.

    Examples
    --------

    >>> hl.eval(hl.qpois(0.99, 1))
    4

    Notes
    -----
    Returns the smallest integer :math:`x` such that Prob(:math:`X \leq x`) :math:`\geq` `p` where :math:`X`
    is a Poisson random variable with rate parameter `lambda`.

    Parameters
    ----------
    p : float or :class:`.Expression` of type :py:data:`.tfloat64`
    lamb : float or :class:`.Expression` of type :py:data:`.tfloat64`
        Rate parameter of Poisson distribution.
    lower_tail : bool or :class:`.BooleanExpression`
        Corresponds to `lower_tail` parameter in inverse :meth:`.ppois`.
    log_p : bool or :class:`.BooleanExpression`
        Exponentiate `p` before testing.

    Returns
    -------
    :class:`.Expression` of type :py:data:`.tfloat64`
    """
    return _func("qpois", tint32, p, lamb, lower_tail, log_p)


@typecheck(start=expr_int32, stop=expr_int32, step=expr_int32)
def range(start, stop, step=1) -> ArrayNumericExpression:
    """Returns an array of integers from `start` to `stop` by `step`.

    Examples
    --------

    >>> hl.eval(hl.range(0, 10))
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

    >>> hl.eval(hl.range(0, 10, step=3))
    [0, 3, 6, 9]

    Notes
    -----
    The range includes `start`, but excludes `stop`.

    Parameters
    ----------
    start : int or :class:`.Expression` of type :py:data:`.tint32`
        Start of range.
    stop : int or :class:`.Expression` of type :py:data:`.tint32`
        End of range.
    step : int or :class:`.Expression` of type :py:data:`.tint32`
        Step of range.

    Returns
    -------
    :class:`.ArrayInt32Expression`
    """
    return apply_expr(lambda sta, sto, ste: ArrayRange(sta, sto, ste), tarray(tint32), start, stop, step)


@typecheck(p=expr_float64, seed=nullable(int))
def rand_bool(p, seed=None) -> BooleanExpression:
    """Returns ``True`` with probability `p`.

    Examples
    --------

    >>> hl.eval(hl.rand_bool(0.5))  # doctest: +NOTEST
    True

    >>> hl.eval(hl.rand_bool(0.5))  # doctest: +NOTEST
    False

    Parameters
    ----------
    p : :obj:`float` or :class:`.Float64Expression`
        Probability between 0 and 1.
    seed : :obj:`int`, optional
        Random seed.

    Returns
    -------
    :class:`.BooleanExpression`
    """
    return _seeded_func("rand_bool", tbool, seed, p)


@typecheck(mean=expr_float64, sd=expr_float64, seed=nullable(int))
def rand_norm(mean=0, sd=1, seed=None) -> Float64Expression:
    """Samples from a normal distribution with mean `mean` and standard
    deviation `sd`.

    Examples
    --------

    >>> hl.eval(hl.rand_norm())  # doctest: +NOTEST
    1.5388475315213386

    >>> hl.eval(hl.rand_norm())  # doctest: +NOTEST
    -0.3006188509144124

    Parameters
    ----------
    mean : :obj:`float` or :class:`.Float64Expression`
        Mean of normal distribution.
    sd : float or :class:`.Expression` of type :py:data:`.tfloat64`
        Standard deviation of normal distribution.
    seed : :obj:`int`, optional
        Random seed.

    Returns
    -------
    :class:`.Float64Expression`
    """
    return _seeded_func("rand_norm", tfloat64, seed, mean, sd)


@typecheck(mean=nullable(expr_array(expr_float64)), cov=nullable(expr_array(expr_float64)), seed=nullable(int))
def rand_norm2d(mean=None, cov=None, seed=None) -> ArrayNumericExpression:
    """Samples from a normal distribution with mean `mean` and covariance matrix `cov`.

    Examples
    --------

    >>> hl.eval(hl.rand_norm2d())  # doctest: +NOTEST
    [0.5515477294463427, -1.1782691532205807]

    >>> hl.eval(hl.rand_norm2d())  # doctest: +NOTEST
    [-1.127240906867922, 1.4495317887283203]

    Notes
    -----
    The covariance of a 2d normal distribution is a 2x2 symmetric matrix
    [[a, b], [b, c]]. This is specified in `cov` as a length 3 array [a, b, c].
    The covariance matrix must be positive semi-definite, i.e. a>0, c>0, and
    a*c - b^2 > 0.

    If `mean` and `cov` are both None, draws from the standard 2d normal
    distribution.

    Parameters
    ----------
    mean : :class:`.ArrayNumericExpression`, optional
        Mean of normal distribution. Array of length 2.
    cov : :class:`.ArrayNumericExpression`, optional
        Covariance of normal distribution. Array of length 3.
    seed : :obj:`int`, optional
        Random seed.

    Returns
    -------
    :class:`.ArrayFloat64Expression`
    """
    if mean is None:
        mean = [0, 0]
    if cov is None:
        cov = [1, 0, 1]

    def f(mean, cov):
        m1 = mean[0]
        m2 = mean[1]
        s11 = cov[0]
        s12 = cov[1]
        s22 = cov[2]

        x = hl.range(0, 2).map(lambda i: rand_norm(seed=seed))
        return hl.rbind(hl.sqrt(s11), (lambda root_s11:
            hl.array([
               m1 + root_s11 * x[0],
               m2 + (s12 / root_s11) * x[0]
                  + hl.sqrt(s22 - s12 * s12 / s11) * x[1]])))

    return hl.rbind(mean, cov, f)


@typecheck(lamb=expr_float64, seed=nullable(int))
def rand_pois(lamb, seed=None) -> Float64Expression:
    """Samples from a `Poisson distribution
    <https://en.wikipedia.org/wiki/Poisson_distribution>`__ with rate parameter
    `lamb`.

    Examples
    --------

    >>> hl.eval(hl.rand_pois(1))  # doctest: +NOTEST
    2.0

    >>> hl.eval(hl.rand_pois(1))  # doctest: +NOTEST
    3.0

    Parameters
    ----------
    lamb :  :obj:`float` or :class:`.Float64Expression`
        Rate parameter for Poisson distribution.
    seed : :obj:`int`, optional
        Random seed.

    Returns
    -------
    :class:`.Float64Expression`
    """
    return _seeded_func("rand_pois", tfloat64, seed, lamb)


@typecheck(lower=expr_float64, upper=expr_float64, seed=nullable(int))
def rand_unif(lower, upper, seed=None) -> Float64Expression:
    """Samples from a uniform distribution within the interval
    [`lower`, `upper`].

    Examples
    --------

    >>> hl.eval(hl.rand_unif(0, 1))  # doctest: +NOTEST
    0.7983073825816226

    >>> hl.eval(hl.rand_unif(0, 1))  # doctest: +NOTEST
    0.5161799497741769

    Parameters
    ----------
    lower : :obj:`float` or :class:`.Float64Expression`
        Left boundary of range.
    upper : :obj:`float` or :class:`.Float64Expression`
        Right boundary of range.
    seed : :obj:`int`, optional
        Random seed.

    Returns
    -------
    :class:`.Float64Expression`
    """
    return _seeded_func("rand_unif", tfloat64, seed, lower, upper)


@typecheck(a=expr_float64,
           b=expr_float64,
           lower=nullable(expr_float64),
           upper=nullable(expr_float64),
           seed=nullable(int))
def rand_beta(a, b, lower=None, upper=None, seed=None) -> Float64Expression:
    """Samples from a `beta distribution
    <https://en.wikipedia.org/wiki/Beta_distribution>`__ with parameters `a`
    (alpha) and `b` (beta).

    Notes
    -----
    The optional parameters `lower` and `upper` represent a truncated beta
    distribution with parameters a and b and support `[lower, upper]`. Draws are
    made via rejection sampling, i.e. returning the first draw from Beta(a,b)
    that falls in range `[lower, upper]`. This procedure may be slow if the
    probability mass of Beta(a,b) over `[lower, upper]` is small.

    Examples
    --------

    >>> hl.eval(hl.rand_beta(0, 1))  # doctest: +NOTEST
    0.6696807666871818

    >>> hl.eval(hl.rand_beta(0, 1))  # doctest: +NOTEST
    0.8512985039011525

    Parameters
    ----------
    a : :obj:`float` or :class:`.Float64Expression`
    b : :obj:`float` or :class:`.Float64Expression`
    lower : :obj:`float` or :class:`.Float64Expression`, optional
        Lower boundary of truncated beta distribution.
    upper : :obj:`float` or :class:`.Float64Expression`, optional
        Upper boundary of truncated beta distribution.
    seed : :obj:`int`, optional
        Random seed.

    Returns
    -------
    :class:`.Float64Expression`
    """
    if lower is None and upper is None:
        return _seeded_func("rand_beta", tfloat64, seed, a, b)
    if lower is None:
        lower = hl.literal(0)
    if upper is None:
        upper = hl.literal(1)

    return _seeded_func("rand_beta", tfloat64, seed, a, b, lower, upper)


@typecheck(shape=expr_float64,
           scale=expr_float64,
           seed=nullable(int))
def rand_gamma(shape, scale, seed=None) -> Float64Expression:
    """Samples from a `gamma distribution
    <https://en.wikipedia.org/wiki/Gamma_distribution>`__
    with parameters `shape` and `scale`.

    Examples
    --------

    >>> hl.eval(hl.rand_gamma(1, 1))  # doctest: +NOTEST
    2.3915947710237537

    >>> hl.eval(hl.rand_gamma(1, 1))  # doctest: +NOTEST
    0.1339939936379711

    Parameters
    ----------
    shape : :obj:`float` or :class:`.Float64Expression`
    scale : :obj:`float` or :class:`.Float64Expression`
    seed : :obj:`int`, optional
        Random seed.

    Returns
    -------
    :class:`.Float64Expression`
    """
    return _seeded_func("rand_gamma", tfloat64, seed, shape, scale)


@typecheck(prob=expr_array(expr_float64),
           seed=nullable(int))
def rand_cat(prob, seed=None) -> Int32Expression:
    """Samples from a `categorical distribution
    <https://en.wikipedia.org/wiki/Categorical_distribution>`__.

    Notes
    -----
    The categories correspond to the indices of `prob`, an unnormalized
    probability mass function. The probability of drawing index ``i`` is
    ``prob[i]/sum(prob)``.

    Warning
    -------
    This function may be slow when the number of categories is large.

    Examples
    --------

    >>> hl.eval(hl.rand_cat([0, 1.7, 2]))  # doctest: +NOTEST
    2

    >>> hl.eval(hl.rand_cat([0, 1.7, 2]))  # doctest: +NOTEST
    1

    Parameters
    ----------
    prob : :obj:`list` of float or :class:`.ArrayExpression` of type :py:data:`.tfloat64`
    seed : :obj:`int` or `None`
        If not `None`, function will be seeded with provided seed.

    Returns
    -------
    :class:`.Int32Expression`
    """
    return _seeded_func("rand_cat", tint32, seed, prob)


@typecheck(a=expr_array(expr_float64),
           seed=nullable(int))
def rand_dirichlet(a, seed=None) -> ArrayExpression:
    """Samples from a `Dirichlet distribution
    <https://en.wikipedia.org/wiki/Dirichlet_distribution>`__.

    Examples
    --------

    >>> hl.eval(hl.rand_dirichlet([1, 1, 1]))  # doctest: +NOTEST
    [0.4630197581640282,0.18207753442497876,0.3549027074109931]

    >>> hl.eval(hl.rand_dirichlet([1, 1, 1]))  # doctest: +NOTEST
    [0.20851948405364765,0.7873859423649898,0.004094573581362475]

    Parameters
    ----------
    a : :obj:`list` of float or :class:`.ArrayExpression` of type :py:data:`.tfloat64`
        Array of non-negative concentration parameters.
    seed : :obj:`int`, optional
        Random seed.

    Returns
    -------
    :class:`.Float64Expression`
    """
    return hl.bind(lambda x: x / hl.sum(x),
                   a.map(lambda p:
                         hl.cond(p == 0.0,
                                 0.0,
                                 hl.rand_gamma(p, 1, seed=seed))))


@typecheck(x=expr_float64)
def sqrt(x) -> Float64Expression:
    """Returns the square root of `x`.

    Examples
    --------

    >>> hl.eval(hl.sqrt(3))
    1.7320508075688772

    Notes
    -----
    It is also possible to exponentiate expression with standard Python syntax,
    e.g. ``x ** 0.5``.

    Parameters
    ----------
    x : float or :class:`.Expression` of type :py:data:`.tfloat64`

    Returns
    -------
    :class:`.Expression` of type :py:data:`.tfloat64`
    """
    return _func("sqrt", tfloat64, x)

@typecheck(x=expr_array(expr_float64), y=expr_array(expr_float64))
def corr(x, y) -> Float64Expression:
    """Compute the
    `Pearson correlation coefficient <https://en.wikipedia.org/wiki/Pearson_correlation_coefficient>`__
    between `x` and `y`.

    Examples
    --------
    >>> hl.eval(hl.corr([1, 2, 4], [2, 3, 1]))  # doctest: +NOTEST
    -0.6546536707079772

    Notes
    -----
    Only indices where both `x` and `y` are non-missing will be included in the
    calculation.

    If `x` and `y` have length zero, then the result is missing.

    Parameters
    ----------
    x : :class:`.Expression` of type ``array<tfloat64>``
    y : :class:`.Expression` of type ``array<tfloat64>``

    Returns
    -------
    :class:`.Float64Expression`
    """
    return _func("corr", tfloat64, x, y)

_base_regex = "^([ACGTNM])+$"
_symbolic_regex = r"(^\.)|(\.$)|(^<)|(>$)|(\[)|(\])"
_allele_types = ["Unknown", "SNP", "MNP", "Insertion", "Deletion", "Complex", "Star", "Symbolic"]
_allele_enum = {i: v for i, v in enumerate(_allele_types)}
_allele_ints = {v: k for k, v in _allele_enum.items()}


@typecheck(ref=expr_str, alt=expr_str)
def _num_allele_type(ref, alt) -> Int32Expression:
    return hl.bind(lambda r, a:
                   hl.cond(r.matches(_base_regex),
                           hl.case()
                           .when(a.matches(_base_regex), hl.case()
                                 .when(r.length() == a.length(),
                                       hl.cond(r.length() == 1,
                                               hl.cond(r != a, _allele_ints['SNP'], _allele_ints['Unknown']),
                                               hl.cond(hamming(r, a) == 1,
                                                       _allele_ints['SNP'],
                                                       _allele_ints['MNP'])))
                                 .when((r.length() < a.length()) & (r[0] == a[0]) & a.endswith(r[1:]),
                                       _allele_ints["Insertion"])
                                 .when((r[0] == a[0]) & r.endswith(a[1:]),
                                       _allele_ints["Deletion"])
                                 .default(_allele_ints['Complex']))
                           .when(a == '*', _allele_ints['Star'])
                           .when(a.matches(_symbolic_regex), _allele_ints['Symbolic'])
                           .default(_allele_ints['Unknown']),
                           _allele_ints['Unknown']),
                   ref, alt)


@typecheck(ref=expr_str, alt=expr_str)
def is_snp(ref, alt) -> BooleanExpression:
    """Returns ``True`` if the alleles constitute a single nucleotide polymorphism.

    Examples
    --------

    >>> hl.eval(hl.is_snp('A', 'T'))
    True

    Parameters
    ----------
    ref : :class:`.StringExpression`
        Reference allele.
    alt : :class:`.StringExpression`
        Alternate allele.

    Returns
    -------
    :class:`.BooleanExpression`
    """
    return _num_allele_type(ref, alt) == _allele_ints["SNP"]


@typecheck(ref=expr_str, alt=expr_str)
def is_mnp(ref, alt) -> BooleanExpression:
    """Returns ``True`` if the alleles constitute a multiple nucleotide polymorphism.

    Examples
    --------

    >>> hl.eval(hl.is_mnp('AA', 'GT'))
    True

    Parameters
    ----------
    ref : :class:`.StringExpression`
        Reference allele.
    alt : :class:`.StringExpression`
        Alternate allele.

    Returns
    -------
    :class:`.BooleanExpression`
    """
    return _num_allele_type(ref, alt) == _allele_ints["MNP"]


@typecheck(ref=expr_str, alt=expr_str)
def is_transition(ref, alt) -> BooleanExpression:
    """Returns ``True`` if the alleles constitute a transition.

    Examples
    --------

    >>> hl.eval(hl.is_transition('A', 'T'))
    False

    >>> hl.eval(hl.is_transition('AAA', 'AGA'))
    True

    Parameters
    ----------
    ref : :class:`.StringExpression`
        Reference allele.
    alt : :class:`.StringExpression`
        Alternate allele.

    Returns
    -------
    :class:`.BooleanExpression`
    """
    return is_snp(ref, alt) & _is_snp_transition(ref, alt)


@typecheck(ref=expr_str, alt=expr_str)
def is_transversion(ref, alt) -> BooleanExpression:
    """Returns ``True`` if the alleles constitute a transversion.

    Examples
    --------

    >>> hl.eval(hl.is_transversion('A', 'T'))
    True

    >>> hl.eval(hl.is_transversion('AAA', 'AGA'))
    False

    Parameters
    ----------
    ref : :class:`.StringExpression`
        Reference allele.
    alt : :class:`.StringExpression`
        Alternate allele.

    Returns
    -------
    :class:`.BooleanExpression`
    """
    return is_snp(ref, alt) & (~(_is_snp_transition(ref, alt)))


@typecheck(ref=expr_str, alt=expr_str)
def _is_snp_transition(ref, alt) -> BooleanExpression:
    indices = hl.range(0, ref.length())
    return hl.any(lambda i: ((ref[i] != alt[i]) & (((ref[i] == 'A') & (alt[i] == 'G')) |
                                                   ((ref[i] == 'G') & (alt[i] == 'A')) |
                                                   ((ref[i] == 'C') & (alt[i] == 'T')) |
                                                   ((ref[i] == 'T') & (alt[i] == 'C')))), indices)

@typecheck(ref=expr_str, alt=expr_str)
def is_insertion(ref, alt) -> BooleanExpression:
    """Returns ``True`` if the alleles constitute an insertion.

    Examples
    --------

    >>> hl.eval(hl.is_insertion('A', 'ATT'))
    True

    Parameters
    ----------
    ref : :class:`.StringExpression`
        Reference allele.
    alt : :class:`.StringExpression`
        Alternate allele.

    Returns
    -------
    :class:`.BooleanExpression`
    """
    return _num_allele_type(ref, alt) == _allele_ints["Insertion"]


@typecheck(ref=expr_str, alt=expr_str)
def is_deletion(ref, alt) -> BooleanExpression:
    """Returns ``True`` if the alleles constitute a deletion.

    Examples
    --------

    >>> hl.eval(hl.is_deletion('ATT', 'A'))
    True

    Parameters
    ----------
    ref : :class:`.StringExpression`
        Reference allele.
    alt : :class:`.StringExpression`
        Alternate allele.

    Returns
    -------
    :class:`.BooleanExpression`
    """
    return _num_allele_type(ref, alt) == _allele_ints["Deletion"]


@typecheck(ref=expr_str, alt=expr_str)
def is_indel(ref, alt) -> BooleanExpression:
    """Returns ``True`` if the alleles constitute an insertion or deletion.

    Examples
    --------

    >>> hl.eval(hl.is_indel('ATT', 'A'))
    True

    Parameters
    ----------
    ref : :class:`.StringExpression`
        Reference allele.
    alt : :class:`.StringExpression`
        Alternate allele.

    Returns
    -------
    :class:`.BooleanExpression`
    """
    return hl.bind(lambda t: (t == _allele_ints["Insertion"]) |
                             (t == _allele_ints["Deletion"]),
                   _num_allele_type(ref, alt))


@typecheck(ref=expr_str, alt=expr_str)
def is_star(ref, alt) -> BooleanExpression:
    """Returns ``True`` if the alleles constitute an upstream deletion.

    Examples
    --------

    >>> hl.eval(hl.is_star('A', '*'))
    True

    Parameters
    ----------
    ref : :class:`.StringExpression`
        Reference allele.
    alt : :class:`.StringExpression`
        Alternate allele.

    Returns
    -------
    :class:`.BooleanExpression`
    """
    return _num_allele_type(ref, alt) == _allele_ints["Star"]


@typecheck(ref=expr_str, alt=expr_str)
def is_complex(ref, alt) -> BooleanExpression:
    """Returns ``True`` if the alleles constitute a complex polymorphism.

    Examples
    --------

    >>> hl.eval(hl.is_complex('ATT', 'GCAC'))
    True

    Parameters
    ----------
    ref : :class:`.StringExpression`
        Reference allele.
    alt : :class:`.StringExpression`
        Alternate allele.

    Returns
    -------
    :class:`.BooleanExpression`
    """
    return _num_allele_type(ref, alt) == _allele_ints["Complex"]


@typecheck(ref=expr_str, alt=expr_str)
def is_strand_ambiguous(ref, alt) -> BooleanExpression:
    """Returns ``True`` if the alleles are strand ambiguous.

    Strand ambiguous allele pairs are ``A/T``, ``T/A``,
    ``C/G``, and ``G/C`` where the first allele is `ref`
    and the second allele is `alt`.

    Examples
    --------

    >>> hl.eval(hl.is_strand_ambiguous('A', 'T'))
    True

    Parameters
    ----------
    ref : :class:`.StringExpression`
        Reference allele.
    alt : :class:`.StringExpression`
        Alternate allele.

    Returns
    -------
    :class:`.BooleanExpression`
    """
    alleles = hl.literal({('A', 'T'), ('T', 'A'), ('G', 'C'), ('C', 'G')})
    return alleles.contains((ref, alt))


@typecheck(ref=expr_str, alt=expr_str)
def allele_type(ref, alt)-> StringExpression:
    """Returns the type of the polymorphism as a string.

    Examples
    --------

    >>> hl.eval(hl.allele_type('A', 'T'))
    'SNP'

    >>> hl.eval(hl.allele_type('ATT', 'A'))
    'Deletion'

    Notes
    -----
    The possible return values are:
     - ``"SNP"``
     - ``"MNP"``
     - ``"Insertion"``
     - ``"Deletion"``
     - ``"Complex"``
     - ``"Star"``
     - ``"Symbolic"``
     - ``"Unknown"``

    Parameters
    ----------
    ref : :class:`.StringExpression`
        Reference allele.
    alt : :class:`.StringExpression`
        Alternate allele.

    Returns
    -------
    :class:`.StringExpression`
    """
    return hl.literal(_allele_types)[_num_allele_type(ref, alt)]


@typecheck(s1=expr_str, s2=expr_str)
def hamming(s1, s2) -> Int32Expression:
    """Returns the Hamming distance between the two strings.

    Examples
    --------

    >>> hl.eval(hl.hamming('ATATA', 'ATGCA'))
    2

    >>> hl.eval(hl.hamming('abcdefg', 'zzcdefz'))
    3

    Notes
    -----
    This method will fail if the two strings have different length.

    Parameters
    ----------
    s1 : :class:`.StringExpression`
        First string.
    s2 : :class:`.StringExpression`
        Second string.

    Returns
    -------
    :class:`.Expression` of type :py:data:`.tint32`
    """
    return _func("hamming", tint32, s1, s2)


@typecheck(s=expr_str)
def entropy(s) -> Float64Expression:
    r"""Returns the `Shannon entropy <https://en.wikipedia.org/wiki/Entropy_(information_theory)>`__
    of the character distribution defined by the string.

    Examples
    --------

    >>> hl.eval(hl.entropy('ac'))
    1.0

    >>> hl.eval(hl.entropy('accctg'))
    1.7924812503605778

    Notes
    -----
    For a string of length :math:`n` with :math:`k` unique characters
    :math:`\{ c_1, \dots, c_k \}`, let :math:`p_i` be the probability that
    a randomly chosen character is :math:`c_i`, e.g. the number of instances
    of :math:`c_i` divided by :math:`n`. Then the base-2 Shannon entropy is
    given by

    .. math::

        H = \sum_{i=1}^k p_i \log_2(p_i).

    Parameters
    ----------
    s : :class:`.StringExpression`

    Returns
    -------
    :class:`.Expression` of type :py:data:`.tfloat64`
    """
    return _func("entropy", tfloat64, s)


@typecheck(x=expr_any)
def str(x) -> StringExpression:
    """Returns the string representation of `x`.

    Examples
    --------

    >>> hl.eval(hl.str(hl.struct(a=5, b=7)))
    '{"a":5,"b":7}'

    Parameters
    ----------
    x

    Returns
    -------
    :class:`.StringExpression`
    """
    if x.dtype == tstr:
        return x
    else:
        return _func("str", tstr, x)


@typecheck(c=expr_call, i=expr_int32)
def downcode(c, i) -> CallExpression:
    """Create a new call by setting all alleles other than i to ref

    Examples
    --------
    Preserve the third allele and downcode all other alleles to reference.

    >>> hl.eval(hl.downcode(hl.call(1, 2), 2))
    Call(alleles=[0, 1], phased=False)

    >>> hl.eval(hl.downcode(hl.call(2, 2), 2))
    Call(alleles=[1, 1], phased=False)

    >>> hl.eval(hl.downcode(hl.call(0, 1), 2))
    Call(alleles=[0, 0], phased=False)

    Parameters
    ----------
    c : :class:`.CallExpression`
        A call.
    i : :class:`.Expression` of type :py:data:`.tint32`
        The index of the allele that will be sent to the alternate allele. All
        other alleles will be downcoded to reference.

    Returns
    -------
    :class:`.CallExpression`
    """
    return _func("downcode", tcall, c, i)


@typecheck(pl=expr_array(expr_int32))
def gq_from_pl(pl) -> Int32Expression:
    """Compute genotype quality from Phred-scaled probability likelihoods.

    Examples
    --------

    >>> hl.eval(hl.gq_from_pl([0, 69, 1035]))
    69

    Parameters
    ----------
    pl : :class:`.ArrayInt32Expression`

    Returns
    -------
    :class:`.Expression` of type :py:data:`.tint32`
    """
    return _func("gqFromPL", tint32, pl)


@typecheck(n=expr_int32)
def triangle(n) -> Int32Expression:
    """Returns the triangle number of `n`.

    Examples
    --------

    >>> hl.eval(hl.triangle(3))
    6

    Notes
    -----
    The calculation is ``n * (n + 1) / 2``.

    Parameters
    ----------
    n : :class:`.Expression` of type :py:data:`.tint32`

    Returns
    -------
    :class:`.Expression` of type :py:data:`.tint32`
    """
    return _func("triangle", tint32, n)


@typecheck(f=func_spec(1, expr_bool),
           collection=expr_oneof(expr_set(), expr_array()))
def filter(f: Callable, collection):
    """Returns a new collection containing elements where `f` returns ``True``.

    Examples
    --------

    >>> a = [1, 2, 3, 4]
    >>> s = {'Alice', 'Bob', 'Charlie'}

    >>> hl.eval(hl.filter(lambda x: x % 2 == 0, a))
    [2, 4]

    >>> hl.eval(hl.filter(lambda x: ~(x[-1] == 'e'), s))
    {'Bob'}

    Notes
    -----
    Returns a same-type expression; evaluated on a :class:`.SetExpression`, returns a
    :class:`.SetExpression`. Evaluated on an :class:`.ArrayExpression`,
    returns an :class:`.ArrayExpression`.

    Parameters
    ----------
    f : function ( (arg) -> :class:`.BooleanExpression`)
        Function to evaluate for each element of the collection. Must return a
        :class:`.BooleanExpression`.
    collection : :class:`.ArrayExpression` or :class:`.SetExpression`.
        Array or set expression to filter.

    Returns
    -------
    :class:`.ArrayExpression` or :class:`.SetExpression`
        Expression of the same type as `collection`.
    """
    return collection.filter(f)


@typecheck(f=func_spec(1, expr_bool),
           collection=expr_oneof(expr_set(), expr_array()))
def any(f: Callable, collection) -> BooleanExpression:
    """Returns ``True`` if `f` returns ``True`` for any element.

    Examples
    --------

    >>> a = ['The', 'quick', 'brown', 'fox']
    >>> s = {1, 3, 5, 6, 7, 9}

    >>> hl.eval(hl.any(lambda x: x[-1] == 'x', a))
    True

    >>> hl.eval(hl.any(lambda x: x % 4 == 0, s))
    False

    Notes
    -----
    This method returns ``False`` for empty collections.

    Parameters
    ----------
    f : function ( (arg) -> :class:`.BooleanExpression`)
        Function to evaluate for each element of the collection. Must return a
        :class:`.BooleanExpression`.
    collection : :class:`.ArrayExpression` or :class:`.SetExpression`
        Collection expression.

    Returns
    -------
    :class:`.BooleanExpression`.
        ``True`` if `f` returns ``True`` for any element, ``False`` otherwise.
    """
    return collection.any(f)


@typecheck(f=func_spec(1, expr_bool),
           collection=expr_oneof(expr_set(), expr_array()))
def all(f: Callable, collection) -> BooleanExpression:
    """Returns ``True`` if `f` returns ``True`` for every element.

    Examples
    --------

    >>> a = ['The', 'quick', 'brown', 'fox']
    >>> s = {1, 3, 5, 6, 7, 9}

    >>> hl.eval(hl.all(lambda x: hl.len(x) > 3, a))
    False

    >>> hl.eval(hl.all(lambda x: x < 10, s))
    True

    Notes
    -----
    This method returns ``True`` if the collection is empty.

    Parameters
    ----------
    f : function ( (arg) -> :class:`.BooleanExpression`)
        Function to evaluate for each element of the collection. Must return a
        :class:`.BooleanExpression`.
    collection : :class:`.ArrayExpression` or :class:`.SetExpression`
        Collection expression.

    Returns
    -------
    :class:`.BooleanExpression`.
        ``True`` if `f` returns ``True`` for every element, ``False`` otherwise.
    """

    return collection.all(f)


@typecheck(f=func_spec(1, expr_bool),
           collection=expr_oneof(expr_set(), expr_array()))
def find(f: Callable, collection):
    """Returns the first element where `f` returns ``True``.

    Examples
    --------

    >>> a = ['The', 'quick', 'brown', 'fox']
    >>> s = {1, 3, 5, 6, 7, 9}

    >>> hl.eval(hl.find(lambda x: x[-1] == 'x', a))
    'fox'

    >>> hl.eval(hl.find(lambda x: x % 4 == 0, s))
    None

    Notes
    -----
    If `f` returns ``False`` for every element, then the result is missing.

    Sets are unordered. If `collection` is of type :class:`.tset`, then the
    element returned comes from no guaranteed ordering.

    Parameters
    ----------
    f : function ( (arg) -> :class:`.BooleanExpression`)
        Function to evaluate for each element of the collection. Must return a
        :class:`.BooleanExpression`.
    collection : :class:`.ArrayExpression` or :class:`.SetExpression`
        Collection expression.

    Returns
    -------
    :class:`.Expression`
        Expression whose type is the element type of the collection.
    """
    return collection.find(f)


@typecheck(f=func_spec(1, expr_any),
           collection=expr_oneof(expr_set(), expr_array()))
def flatmap(f: Callable, collection):
    """Map each element of the collection to a new collection, and flatten the results.

    Examples
    --------

    >>> a = [[0, 1], [1, 2], [4, 5, 6, 7]]

    >>> hl.eval(hl.flatmap(lambda x: x[1:], a))
    [1, 2, 5, 6, 7]

    Parameters
    ----------
    f : function ( (arg) -> :class:`.CollectionExpression`)
        Function from the element type of the collection to the type of the
        collection. For instance, `flatmap` on a ``set<str>`` should take
        a ``str`` and return a ``set``.
    collection : :class:`.ArrayExpression` or :class:`.SetExpression`
        Collection expression.

    Returns
    -------
    :class:`.ArrayExpression` or :class:`.SetExpression`
    """
    expected_type, s = (tarray, 'Array') if isinstance(collection.dtype, tarray) else (tset, 'Set')

    def unify_ret(t):
        if not isinstance(t, expected_type):
            raise TypeError("'flatmap' expects 'f' to return an expression of type '{}', found '{}'".format(s, t))
        return t

    return collection.flatmap(f)


@typecheck(f=func_spec(1, expr_any),
           collection=expr_oneof(expr_set(), expr_array()))
def group_by(f: Callable, collection) -> DictExpression:
    """Group collection elements into a dict according to a lambda function.

    Examples
    --------

    >>> a = ['The', 'quick', 'brown', 'fox']

    >>> hl.eval(hl.group_by(lambda x: hl.len(x), a))  # doctest: +NOTEST
    {5: ['quick', 'brown'], 3: ['The', 'fox']}

    Parameters
    ----------
    f : function ( (arg) -> :class:`.Expression`)
        Function to evaluate for each element of the collection to produce a key for the
        resulting dictionary.
    collection : :class:`.ArrayExpression` or :class:`.SetExpression`
        Collection expression.

    Returns
    -------
    :class:`.DictExpression`.
        Dictionary keyed by results of `f`.
    """
    return collection.group_by(f)


@typecheck(f=func_spec(2, expr_any),
           zero=expr_any,
           collection=expr_oneof(expr_set(), expr_array()))
def fold(f: Callable, zero, collection) -> Expression:
    """Reduces a collection with the given function `f`, provided the initial value `zero`.

    Examples
    --------
    >>> a = [0, 1, 2]

    >>> hl.eval(hl.fold(lambda i, j: i + j, 0, a))
    3

    Parameters
    ----------
    f : function ( (:class:`.Expression`, :class:`.Expression`) -> :class:`.Expression`)
        Function which takes the cumulative value and the next element, and
        returns a new value.
    zero : :class:`.Expression`
        Initial value to pass in as left argument of `f`.
    collection : :class:`.ArrayExpression` or :class:`.SetExpression`

    Returns
    -------
    :class:`.Expression`
    """
    return collection.fold(lambda x, y: f(x, y), zero)


@typecheck(f=func_spec(2, expr_any),
           zero=expr_any,
           a=expr_array())
def array_scan(f: Callable, zero, a) -> ArrayExpression:
    """Map each element of `a` to cumulative value of function `f`, with initial value `zero`.

    Examples
    --------
    >>> a = [0, 1, 2]

    >>> hl.eval(hl.array_scan(lambda i, j: i + j, 0, a))
    [0, 0, 1, 3]

    Parameters
    ----------
    f : function ( (:class:`.Expression`, :class:`.Expression`) -> :class:`.Expression`)
        Function which takes the cumulative value and the next element, and
        returns a new value.
    zero : :class:`.Expression`
        Initial value to pass in as left argument of `f`.
    a : :class:`.ArrayExpression`

    Returns
    -------
    :class:`.ArrayExpression`.
    """
    return a.scan(lambda x, y: f(x, y), zero)


@typecheck(arrays=expr_array(), fill_missing=bool)
def zip(*arrays, fill_missing: bool = False) -> ArrayExpression:
    """Zip together arrays into a single array.

    Examples
    --------

    >>> hl.eval(hl.zip([1], [10, 20], [100, 200, 300]))
    [(1, 10, 100)]

    >>> hl.eval(hl.zip([1], [10, 20], [100, 200, 300], fill_missing=True))
    [(1, 10, 100), (None, 20, 200), (None, None, 300)]

    Notes
    -----
    The element type of the resulting array is a :class:`.ttuple` with a field
    for each array.

    Parameters
    ----------
    arrays: : variable-length args of :class:`.ArrayExpression`
        Array expressions.
    fill_missing : :obj:`bool`
        If ``False``, return an array with length equal to the shortest length
        of the `arrays`. If ``True``, return an array equal to the longest
        length of the `arrays`, by extending the shorter arrays with missing
        values.

    Returns
    -------
    :class:`.ArrayExpression`
    """

    n_arrays = builtins.len(arrays)
    if fill_missing:
        def _(array_lens):
            result_len = hl.max(array_lens)
            indices = hl.range(0, result_len)
            return hl.map(lambda i: builtins.tuple(
                hl.cond(i < array_lens[j], arrays[j][i], hl.null(arrays[j].dtype.element_type))
                for j in builtins.range(n_arrays)), indices)

        return bind(_, [hl.len(a) for a in arrays])
    else:
        def _(array_lens):
            result_len = hl.min(array_lens)
            indices = hl.range(0, result_len)
            return hl.map(lambda i: builtins.tuple(arrays[j][i] for j in builtins.range(n_arrays)), indices)

        return bind(_, [hl.len(a) for a in arrays])


@typecheck(a=expr_array(), index_first=bool)
def zip_with_index(a, index_first=True):
    """Returns an array of (index, element) tuples.

    Examples
    --------

    >>> hl.eval(hl.zip_with_index(['A', 'B', 'C']))
    [(0, 'A'), (1, 'B'), (2, 'C')]

    >>> hl.eval(hl.zip_with_index(['A', 'B', 'C'], index_first=False))
    [('A', 0), ('B', 1), ('C', 2)]


    Parameters
    ----------
    a : :class:`.ArrayExpression`
    index_first: :obj:`bool`
        If ``True``, the index is the first value of the element tuples. If
        ``False``, the index is the second value.

    Returns
    -------
    :class:`.ArrayExpression`
        Array of (index, element) or (element, index) tuples.
    """
    return bind(lambda aa: range(0, len(aa)).map(lambda i: (i, aa[i]) if index_first else (aa[i], i)), a)


@typecheck(f=func_spec(1, expr_any),
           collection=expr_oneof(expr_set(), expr_array(), expr_ndarray()))
def map(f: Callable, collection):
    """Transform each element of a collection.

    Examples
    --------

    >>> a = ['The', 'quick', 'brown', 'fox']

    >>> hl.eval(hl.map(lambda x: hl.len(x), a))
    [3, 5, 5, 3]

    Parameters
    ----------
    f : function ( (arg) -> :class:`.Expression`)
        Function to transform each element of the collection.
    collection : :class:`.ArrayExpression` or :class:`.SetExpression`
        Collection expression.

    Returns
    -------
    :class:`.ArrayExpression` or :class:`SetExpression`.
        Collection where each element has been transformed by `f`.
    """
    return collection.map(f)


@typecheck(x=expr_oneof(expr_set(), expr_array(), expr_dict(), expr_str, expr_tuple(), expr_struct()))
def len(x) -> Int32Expression:
    """Returns the size of a collection or string.

    Examples
    --------

    >>> a = ['The', 'quick', 'brown', 'fox']
    >>> s = {1, 3, 5, 6, 7, 9}

    >>> hl.eval(hl.len(a))
    4

    >>> hl.eval(hl.len(s))
    6

    >>> hl.eval(hl.len("12345"))
    5

    Parameters
    ----------
    x : :class:`.ArrayExpression` or :class:`.SetExpression` or :class:`.DictExpression` or :class:`.StringExpression`
        String or collection expression.

    Returns
    -------
    :class:`.Expression` of type :py:data:`.tint32`
    """
    if isinstance(x.dtype, ttuple) or isinstance(x.dtype, tstruct):
        return hl.int32(builtins.len(x))
    elif x.dtype == tstr:
        return apply_expr(lambda x: Apply("length", x), tint32, x)
    else:
        return apply_expr(lambda x: ArrayLen(x), tint32, array(x))


@typecheck(x=expr_oneof(expr_array(), expr_str))
def reversed(x):
    """Reverses the elements of a collection.

    Examples
    --------
    >>> a = ['The', 'quick', 'brown', 'fox']
    >>> hl.eval(hl.reversed(a))
    ['fox', 'brown', 'quick', 'The']

    Parameters
    ----------
    x : :class:`.ArrayExpression` or :class:`.StringExpression`
        Array or string expression.

    Returns
    -------
    :class:`.Expression`
    """

    typ = x.dtype
    x = range(0, len(x)).map(lambda i: x[len(x) - 1 - i])
    if typ == tstr:
        x = hl.delimit(x, '')
    return x


@typecheck(exprs=expr_oneof(expr_numeric, expr_set(expr_numeric), expr_array(expr_numeric)),
           filter_missing=bool)
def max(*exprs, filter_missing: bool = True) -> NumericExpression:
    """Returns the maximum element of a collection or of given numeric expressions.

    Examples
    --------

    Take the maximum value of an array:

    >>> hl.eval(hl.max([1, 3, 5, 6, 7, 9]))
    9

    Take the maximum value of values:

    >>> hl.eval(hl.max(1, 50, 2))
    50

    Notes
    -----
    Like the Python builtin ``max`` function, this function can either take a
    single iterable expression (an array or set of numeric elements), or
    variable-length arguments of numeric expressions.

    Note
    ----
    Missing arguments / array elements are ignored if `filter_missing` is ``True``.
    If `filter_missing` is ``False``, then any missing element causes the result to
    be missing.

    Parameters
    ----------
    exprs : :class:`.ArrayExpression` or :class:`.SetExpression` or varargs of :class:`.NumericExpression`
        Single numeric array or set, or multiple numeric values.
    filter_missing : :obj:`bool`
        Remove missing elements from the collection before computing product.

    Returns
    -------
    :class:`.NumericExpression`
    """
    if builtins.len(exprs) < 1:
        raise ValueError("'max' requires at least one argument")
    if builtins.len(exprs) == 1:
        expr = exprs[0]
        if not (isinstance(expr.dtype, tset) or isinstance(expr.dtype, tarray)):
            raise TypeError("'max' expects a single numeric array expression or multiple numeric expressions\n"
                            "  Found 1 argument of type '{}'".format(expr.dtype))
        return expr._filter_missing_method(filter_missing, 'max', expr.dtype.element_type)
    else:
        if not builtins.all(is_numeric(e.dtype) for e in exprs):
            raise TypeError("'max' expects a single numeric array expression or multiple numeric expressions\n"
                            "  Found {} arguments with types '{}'".format(builtins.len(exprs), ', '.join(
                "'{}'".format(e.dtype) for e in exprs)))
        return max(hl.array(list(exprs)), filter_missing=filter_missing)


@typecheck(exprs=expr_oneof(expr_numeric, expr_set(expr_numeric), expr_array(expr_numeric)),
           filter_missing=bool)
def min(*exprs, filter_missing: bool = True) -> NumericExpression:
    """Returns the minimum of a collection or of given numeric expressions.

    Examples
    --------

    Take the minimum value of an array:

    >>> hl.eval(hl.min([2, 3, 5, 6, 7, 9]))
    2

    Take the minimum value:

    >>> hl.eval(hl.min(12, 50, 2))
    2

    Notes
    -----
    Like the Python builtin ``min`` function, this function can either take a
    single iterable expression (an array or set of numeric elements), or
    variable-length arguments of numeric expressions.

    Note
    ----
    Missing arguments / array elements are ignored if `filter_missing` is ``True``.
    If `filter_missing` is ``False``, then any missing element causes the result to
    be missing.

    Parameters
    ----------
    exprs : :class:`.ArrayExpression` or :class:`.SetExpression` or varargs of :class:`.NumericExpression`
        Single numeric array or set, or multiple numeric values.
    filter_missing : :obj:`bool`
        Remove missing elements from the collection before computing product.

    Returns
    -------
    :class:`.NumericExpression`
    """
    if builtins.len(exprs) < 1:
        raise ValueError("'min' requires at least one argument")
    if builtins.len(exprs) == 1:
        expr = exprs[0]
        if not (isinstance(expr.dtype, tset) or isinstance(expr.dtype, tarray)):
            raise TypeError("'min' expects a single numeric array expression or multiple numeric expressions\n"
                            "  Found 1 argument of type '{}'".format(expr.dtype))
        return expr._filter_missing_method(filter_missing, 'min', expr.dtype.element_type)
    else:
        if not builtins.all(is_numeric(e.dtype) for e in exprs):
            raise TypeError("'min' expects a single numeric array expression or multiple numeric expressions\n"
                            "  Found {} arguments with types '{}'".format(builtins.len(exprs), ', '.join(
                "'{}'".format(e.dtype) for e in exprs)))
        return min(hl.array(list(exprs)), filter_missing=filter_missing)


@typecheck(x=expr_oneof(expr_numeric, expr_array(expr_numeric), expr_ndarray(expr_numeric)))
def abs(x):
    """Take the absolute value of a numeric value, array or ndarray.

    Examples
    --------

    >>> hl.eval(hl.abs(-5))
    5

    >>> hl.eval(hl.abs([1.0, -2.5, -5.1]))
    [1.0, 2.5, 5.1]

    Parameters
    ----------
    x : :class:`.NumericExpression`, :class:`.ArrayNumericExpression` or :class:`.NDArrayNumericExpression`

    Returns
    -------
    :class:`.NumericExpression`, :class:`.ArrayNumericExpression` or :class:`.NDArrayNumericExpression`.
    """
    if isinstance(x.dtype, tarray) or isinstance(x.dtype, tndarray):
        return map(abs, x)
    else:
        return x._method('abs', x.dtype)


@typecheck(x=expr_oneof(expr_numeric, expr_array(expr_numeric), expr_ndarray(expr_numeric)))
def sign(x):
    """Returns the sign of a numeric value, array or ndarray.

    Examples
    --------

    >>> hl.eval(hl.sign(-1.23))
    -1.0

    >>> hl.eval(hl.sign([-4, 0, 5]))
    [-1, 0, 1]

    >>> hl.eval(hl.sign([0.0, 3.14]))
    [0.0, 1.0]

    >>> hl.eval(hl.sign(float('nan')))  # doctest: +SKIP
    nan

    Notes
    -----
    The sign function preserves type and maps ``nan`` to ``nan``.

    Parameters
    ----------
    x : :class:`.NumericExpression`, :class:`.ArrayNumericExpression` or :class:`.NDArrayNumericExpression`

    Returns
    -------
    :class:`.NumericExpression`, :class:`.ArrayNumericExpression` or :class:`.NDArrayNumericExpression`.
    """
    if isinstance(x.dtype, tarray) or isinstance(x.dtype, tndarray):
        return map(sign, x)
    else:
        return x._method('sign', x.dtype)


@typecheck(collection=expr_oneof(expr_set(expr_numeric), expr_array(expr_numeric)),
           filter_missing=bool)
def mean(collection, filter_missing: bool = True) -> Float64Expression:
    """Returns the mean of all values in the collection.

    Examples
    --------

    >>> a = [1, 3, 5, 6, 7, 9]

    >>> hl.eval(hl.mean(a))
    5.166666666666667

    Note
    ----
    Missing elements are ignored if `filter_missing` is ``True``. If `filter_missing`
    is ``False``, then any missing element causes the result to be missing.

    Parameters
    ----------
    collection : :class:`.ArrayExpression` or :class:`.SetExpression`
        Collection expression with numeric element type.
    filter_missing : :obj:`bool`
        Remove missing elements from the collection before computing product.

    Returns
    -------
    :class:`.Expression` of type :py:data:`.tfloat64`
    """
    return collection._filter_missing_method(filter_missing, "mean", tfloat64)


@typecheck(collection=expr_oneof(expr_set(expr_numeric), expr_array(expr_numeric)))
def median(collection) -> NumericExpression:
    """Returns the median value in the collection.

    Examples
    --------

    >>> a = [1, 3, 5, 6, 7, 9]

    >>> hl.eval(hl.median(a))
    5

    Note
    ----
    Missing elements are ignored.

    Parameters
    ----------
    collection : :class:`.ArrayExpression` or :class:`.SetExpression`
        Collection expression with numeric element type.

    Returns
    -------
    :class:`.NumericExpression`
    """
    return collection._method("median", collection.dtype.element_type)


@typecheck(collection=expr_oneof(expr_set(expr_numeric), expr_array(expr_numeric)),
           filter_missing=bool)
def product(collection, filter_missing: bool = True) -> NumericExpression:
    """Returns the product of values in the collection.

    Examples
    --------

    >>> a = [1, 3, 5, 6, 7, 9]

    >>> hl.eval(hl.product(a))
    5670

    Note
    ----
    Missing elements are ignored if `filter_missing` is ``True``. If `filter_missing`
    is ``False``, then any missing element causes the result to be missing.

    Parameters
    ----------
    collection : :class:`.ArrayExpression` or :class:`.SetExpression`
        Collection expression with numeric element type.
    filter_missing : :obj:`bool`
        Remove missing elements from the collection before computing product.

    Returns
    -------
    :class:`.NumericExpression`
    """
    return collection._filter_missing_method(filter_missing, "product", collection.dtype.element_type)


@typecheck(collection=expr_oneof(expr_set(expr_numeric), expr_array(expr_numeric)),
           filter_missing=bool)
def sum(collection, filter_missing: bool = True) -> NumericExpression:
    """Returns the sum of values in the collection.

    Examples
    --------
    >>> a = [1, 3, 5, 6, 7, 9]

    >>> hl.eval(hl.sum(a))
    31

    Note
    ----
    Missing elements are ignored if `filter_missing` is ``True``. If `filter_missing`
    is ``False``, then any missing element causes the result to be missing.

    Parameters
    ----------
    collection : :class:`.ArrayExpression` or :class:`.SetExpression`
        Collection expression with numeric element type.
    filter_missing : :obj:`bool`
        Remove missing elements from the collection before computing product.

    Returns
    -------
    :class:`.NumericExpression`
    """
    return collection._filter_missing_method(filter_missing, "sum", collection.dtype.element_type)


@typecheck(a=expr_array(expr_numeric),
            filter_missing=bool)
def cumulative_sum(a, filter_missing: bool = True) -> ArrayNumericExpression:
    """Returns an array of the cumulative sum of values in the array.

    Examples
    --------
    >>> a = [1, 3, 5, 6, 7, 9]

    >>> hl.eval(hl.cumulative_sum(a))
    [1, 4, 9, 15, 22, 31]

    Note
    ----
    Missing elements are ignored if `filter_missing` is ``True``. If `filter_missing`
    is ``False``, then any missing element causes the result to be missing.

    Parameters
    ----------
    a : :class:`.ArrayNumericExpression`
        Array expression with numeric element type.
    filter_missing : :obj:`bool`
        Remove missing elements from the collection before computing product.

    Returns
    -------
    :class:`.ArrayNumericExpression`
    """
    if filter_missing:
        a = a.filter(hl.is_defined)
    return a.scan(lambda accum, elt: accum + elt, 0)[1:]


@typecheck(kwargs=expr_any)
def struct(**kwargs) -> StructExpression:
    """Construct a struct expression.

    Examples
    --------

    >>> s = hl.struct(a=5, b='Foo')
    >>> hl.eval(s.a)
    5

    Returns
    -------
    :class:`.StructExpression`
        Keyword arguments as a struct.
    """
    return StructExpression._from_fields(kwargs)


def tuple(iterable: Iterable) -> TupleExpression:
    """Construct a tuple expression.

    Examples
    --------

    >>> t = hl.tuple([1, 2, '3'])
    >>> hl.eval(t)
    (1, 2, '3')

    >>> hl.eval(t[2])
    '3'

    Parameters
    ----------
    args : :obj:`Iterable` of :class:`.Expression`
        Tuple elements.

    Returns
    -------
    :class:`.TupleExpression`
    """
    t = builtins.tuple(iterable)
    return to_expr(t)


@typecheck(collection=expr_oneof(expr_set(), expr_array()))
def set(collection) -> SetExpression:
    """Convert a set expression.

    Examples
    --------

    >>> s = hl.set(['Bob', 'Charlie', 'Alice', 'Bob', 'Bob'])
    >>> hl.eval(s)  # doctest: +NOTEST
    {'Alice', 'Bob', 'Charlie'}

    Returns
    -------
    :class:`.SetExpression`
        Set of all unique elements.
    """
    if isinstance(collection.dtype, tset):
        return collection
    return apply_expr(lambda c: ToSet(c), tset(collection.dtype.element_type), collection)


@typecheck(t=hail_type)
def empty_set(t: Union[HailType, str]) -> SetExpression:
    """Returns an empty set of elements of a type `t`.

    Examples
    --------

    >>> hl.eval(hl.empty_set(hl.tstr))
    set()

    Parameters
    ----------
    t : :obj:`str` or :class:`.HailType`
        Type of the set elements.

    Returns
    -------
    :class:`.SetExpression`
    """
    return hl.set(empty_array(t))


@typecheck(collection=expr_oneof(expr_set(), expr_array(), expr_dict()))
def array(collection) -> ArrayExpression:
    """Construct an array expression.

    Examples
    --------

    >>> s = {'Bob', 'Charlie', 'Alice'}

    >>> hl.eval(hl.array(s))
    ['Alice', 'Bob', 'Charlie']

    Parameters
    ----------
    collection : :class:`.ArrayExpression` or :class:`.SetExpression` or :class:`.DictExpression`

    Returns
    -------
    :class:`.ArrayExpression`
    """
    if isinstance(collection.dtype, tarray):
        return collection
    elif isinstance(collection.dtype, tset):
        return apply_expr(lambda c: ToArray(c), tarray(collection.dtype.element_type), collection)
    else:
        assert isinstance(collection.dtype, tdict)
        return _func('dictToArray', tarray(ttuple(collection.dtype.key_type, collection.dtype.value_type)), collection)


@typecheck(t=hail_type)
def empty_array(t: Union[HailType, str]) -> ArrayExpression:
    """Returns an empty array of elements of a type `t`.

    Examples
    --------

    >>> hl.eval(hl.empty_array(hl.tint32))
    []

    Parameters
    ----------
    t : :obj:`str` or :class:`.HailType`
        Type of the array elements.

    Returns
    -------
    :class:`.ArrayExpression`
    """
    array_t = hl.tarray(t)
    ir = MakeArray([], array_t)
    return construct_expr(ir, array_t)


def _ndarray(collection, row_major=True):
    def list_shape(x):
        if isinstance(x, list):
            dim_len = builtins.len(x)
            first, rest = x[0], x[1:]
            inner_shape = list_shape(first)
            for e in rest:
                other_inner_shape = list_shape(e)
                if inner_shape != other_inner_shape:
                    raise ValueError(f'inner dimensions do not match: {inner_shape}, {other_inner_shape}')
            return [dim_len] + inner_shape
        else:
            return []

    def deep_flatten(l):
        result = []
        for e in l:
            if isinstance(e, list):
                result.extend(deep_flatten(e))
            else:
                result.append(e)

        return result

    if isinstance(collection, np.ndarray):
        nd = collection.astype(np.float64)
        data = list(nd.flat)
        shape = nd.shape
    elif isinstance(collection, list):
        shape = list_shape(collection)
        data = deep_flatten(collection)
    else:
        shape = []
        data = hl.array([collection])

    shape_expr = to_expr(shape, ir.tarray(ir.tint64))
    data_expr = hl.array(data)

    ndir = ir.MakeNDArray(builtins.len(shape), data_expr._ir, shape_expr._ir, hl.bool(row_major)._ir)
    return construct_expr(ndir, ndir.typ)


@typecheck(key_type=hail_type, value_type=hail_type)
def empty_dict(key_type: Union[HailType, str], value_type: Union[HailType, str]) -> DictExpression:
    """Returns an empty dictionary with key type `key_type` and value type
    `value_type`.

    Examples
    --------

    >>> hl.eval(hl.empty_dict(hl.tstr, hl.tint32))
    {}

    Parameters
    ----------
    key_type : :obj:`str` or :class:`.HailType`
        Type of the keys.
    value_type : :obj:`str` or :class:`.HailType`
        Type of the values.
    Returns
    -------
    :class:`.DictExpression`
    """
    return hl.dict(hl.empty_array(hl.ttuple(key_type, value_type)))


@typecheck(collection=expr_oneof(expr_set(expr_set()), expr_array(expr_array())))
def flatten(collection):
    """Flatten a nested collection by concatenating sub-collections.

    Examples
    --------

    >>> a = [[1, 2], [2, 3]]

    >>> hl.eval(hl.flatten(a))
    [1, 2, 2, 3]

    Parameters
    ----------
    collection : :class:`.ArrayExpression` or :class:`.SetExpression`
        Collection with element type :class:`.tarray` or :class:`.tset`.

    Returns
    -------
    collection : :class:`.ArrayExpression` or :class:`.SetExpression`
    """
    return collection.flatmap(lambda x: x)


@typecheck(collection=expr_oneof(expr_array(), expr_set()),
           delimiter=expr_str)
def delimit(collection, delimiter=',') -> StringExpression:
    """Joins elements of `collection` into single string delimited by `delimiter`.

    Examples
    --------

    >>> a = ['Bob', 'Charlie', 'Alice', 'Bob', 'Bob']

    >>> hl.eval(hl.delimit(a))
    'Bob,Charlie,Alice,Bob,Bob'

    Notes
    -----
    If the element type of `collection` is not :py:data:`.tstr`, then the
    :func:`str` function will be called on each element before joining with
    the delimiter.

    Parameters
    ----------
    collection : :class:`.ArrayExpression` or :class:`.SetExpression`
        Collection.
    delimiter : str or :class:`.StringExpression`
        Field delimiter.

    Returns
    -------
    :class:`.StringExpression`
        Joined string expression.
    """
    if not collection.dtype.element_type == tstr:
        collection = map(str, collection)
    return collection._method("mkString", tstr, delimiter)


@typecheck(left=expr_any, right=expr_any)
def _compare(left, right):
    if left.dtype != right.dtype:
        raise TypeError(f"'compare' expected 'left' and 'right' to have the same type: found {left.dtype} vs {right.dtype}")
    indices, aggregations = unify_all(left, right)
    return construct_expr(ApplyComparisonOp("Compare", left._ir, right._ir), tint32, indices, aggregations)


@typecheck(collection=expr_array(),
           less_than=nullable(func_spec(2, expr_bool)))
def _sort_by(collection, less_than):
    l = Env.get_uid()
    r = Env.get_uid()
    left = construct_expr(Ref(l), collection.dtype.element_type, collection._indices, collection._aggregations)
    right = construct_expr(Ref(r), collection.dtype.element_type, collection._indices, collection._aggregations)
    return construct_expr(
        ArraySort(collection._ir, l, r, less_than(left, right)._ir),
        collection.dtype,
        collection._indices,
        collection._aggregations)


@typecheck(collection=expr_array(),
           key=nullable(func_spec(1, expr_any)),
           reverse=expr_bool)
def sorted(collection,
           key: Optional[Callable]=None,
           reverse=False) -> ArrayExpression:
    """Returns a sorted array.

    Examples
    --------

    >>> a = ['Charlie', 'Alice', 'Bob']

    >>> hl.eval(hl.sorted(a))
    ['Alice', 'Bob', 'Charlie']

    >>> hl.eval(hl.sorted(a, reverse=True))
    ['Charlie', 'Bob', 'Alice']

    >>> hl.eval(hl.sorted(a, key=lambda x: hl.len(x)))
    ['Bob', 'Alice', 'Charlie']

    Notes
    -----
    The ordered types are :py:data:`.tstr` and numeric types.

    Parameters
    ----------
    collection : :class:`.ArrayExpression`
        Array to sort.
    key: function ( (arg) -> :class:`.Expression`), optional
        Function to evaluate for each element to compute sort key.
    reverse : :class:`.BooleanExpression`
        Sort in descending order.

    Returns
    -------
    :class:`.ArrayExpression`
        Sorted array.
    """

    def comp(l, r):
        return (hl.case()
                .when(hl.is_missing(l), False)
                .when(hl.is_missing(r), True)
                .when(reverse, hl._compare(r, l) < 0)
                .default(hl._compare(l, r) < 0))

    if key is None:
        return _sort_by(collection, comp)
    else:
        with_key = collection.map(lambda elt: hl.tuple([key(elt), elt]))
        return _sort_by(with_key, lambda l, r: comp(l[0], r[0])).map(lambda elt: elt[1])


@typecheck(array=expr_array(expr_numeric), unique=bool)
def argmin(array, unique: bool = False) -> Int32Expression:
    """Return the index of the minimum value in the array.

    Examples
    --------

    >>> hl.eval(hl.argmin([0.2, 0.3, 0.6]))
    0

    >>> hl.eval(hl.argmin([0.4, 0.2, 0.2]))
    1

    >>> hl.eval(hl.argmin([0.4, 0.2, 0.2], unique=True))
    None

    Notes
    -----
    Returns the index of the minimum value in the array.

    If two or more elements are tied for minimum, then the `unique` parameter
    will determine the result. If `unique` is ``False``, then the first index
    will be returned. If `unique` is ``True``, then the result is missing.

    If the array is empty, then the result is missing.

    Note
    ----
    Missing elements are ignored.

    Parameters
    ----------
    array : :class:`.ArrayNumericExpression`
    unique : bool

    Returns
    -------
    :class:`.Expression` of type :py:data:`.tint32`
    """
    if unique:
        return array._method("uniqueMinIndex", tint32)
    else:
        return array._method("argmin", tint32)


@typecheck(array=expr_array(expr_numeric), unique=bool)
def argmax(array, unique: bool = False) -> Int32Expression:
    """Return the index of the maximum value in the array.

    Examples
    --------

    >>> hl.eval(hl.argmax([0.2, 0.2, 0.6]))
    2

    >>> hl.eval(hl.argmax([0.4, 0.4, 0.2]))
    0

    >>> hl.eval(hl.argmax([0.4, 0.4, 0.2], unique=True))
    None

    Notes
    -----
    Returns the index of the maximum value in the array.

    If two or more elements are tied for maximum, then the `unique` parameter
    will determine the result. If `unique` is ``False``, then the first index
    will be returned. If `unique` is ``True``, then the result is missing.

    If the array is empty, then the result is missing.

    Note
    ----
    Missing elements are ignored.

    Parameters
    ----------
    array : :class:`.ArrayNumericExpression`
    unique: bool

    Returns
    -------
    :class:`.Expression` of type :py:data:`.tint32`
    """
    if unique:
        return array._method("uniqueMaxIndex", tint32)
    else:
        return array._method("argmax", tint32)


@typecheck(x=expr_oneof(expr_numeric, expr_bool, expr_str))
def float64(x) -> Float64Expression:
    """Convert to a 64-bit floating point expression.

    Examples
    --------

    >>> hl.eval(hl.float64('1.1'))  # doctest: +NOTEST
    1.1

    >>> hl.eval(hl.float64(1))  # doctest: +NOTEST
    1.0

    >>> hl.eval(hl.float64(True))  # doctest: +NOTEST
    1.0

    Parameters
    ----------
    x : :class:`.NumericExpression` or :class:`.BooleanExpression` or :class:`.StringExpression`

    Returns
    -------
    :class:`.NumericExpression` of type :py:data:`.tfloat64`
    """
    if x.dtype == tfloat64:
        return x
    else:
        return x._method("toFloat64", tfloat64)

@typecheck(x=expr_oneof(expr_numeric, expr_bool, expr_str))
def float32(x) -> Float32Expression:
    """Convert to a 32-bit floating point expression.

    Examples
    --------

    >>> hl.eval(hl.float32('1.1'))  # doctest: +NOTEST
    1.1

    >>> hl.eval(hl.float32(1))  # doctest: +NOTEST
    1.0

    >>> hl.eval(hl.float32(True))  # doctest: +NOTEST
    1.0

    Parameters
    ----------
    x : :class:`.NumericExpression` or :class:`.BooleanExpression` or :class:`.StringExpression`

    Returns
    -------
    :class:`.NumericExpression` of type :py:data:`.tfloat32`
    """
    if x.dtype == tfloat32:
        return x
    else:
        return x._method("toFloat32", tfloat32)

@typecheck(x=expr_oneof(expr_numeric, expr_bool, expr_str))
def int64(x) -> Int64Expression:
    """Convert to a 64-bit integer expression.

    Examples
    --------

    >>> hl.eval(hl.int64('1'))
    1

    >>> hl.eval(hl.int64(1.5))
    1

    >>> hl.eval(hl.int64(True))
    1

    Parameters
    ----------
    x : :class:`.NumericExpression` or :class:`.BooleanExpression` or :class:`.StringExpression`

    Returns
    -------
    :class:`.NumericExpression` of type :py:data:`.tint64`
    """
    if x.dtype == tint64:
        return x
    else:
        return x._method("toInt64", tint64)


@typecheck(x=expr_oneof(expr_numeric, expr_bool, expr_str))
def int32(x) -> Int32Expression:
    """Convert to a 32-bit integer expression.

    Examples
    --------

    >>> hl.eval(hl.int32('1'))
    1

    >>> hl.eval(hl.int32(1.5))
    1

    >>> hl.eval(hl.int32(True))
    1

    Parameters
    ----------
    x : :class:`.NumericExpression` or :class:`.BooleanExpression` or :class:`.StringExpression`

    Returns
    -------
    :class:`.NumericExpression` of type :py:data:`.tint32`
    """
    if x.dtype == tint32:
        return x
    else:
        return x._method("toInt32", tint32)

@typecheck(x=expr_oneof(expr_numeric, expr_bool, expr_str))
def int(x) -> Int32Expression:
    """Convert to a 32-bit integer expression.

    Examples
    --------

    >>> hl.eval(hl.int('1'))
    1

    >>> hl.eval(hl.int(1.5))
    1

    >>> hl.eval(hl.int(True))
    1

    Note
    ----
    Alias for :func:`.int32`.

    Parameters
    ----------
    x : :class:`.NumericExpression` or :class:`.BooleanExpression` or :class:`.StringExpression`

    Returns
    -------
    :class:`.NumericExpression` of type :py:data:`.tint32`
    """
    return int32(x)


@typecheck(x=expr_oneof(expr_numeric, expr_bool, expr_str))
def float(x) -> Float64Expression:
    """Convert to a 64-bit floating point expression.

    Examples
    --------

    >>> hl.eval(hl.float('1.1'))
    1.1

    >>> hl.eval(hl.float(1))
    1.0

    >>> hl.eval(hl.float(True))
    1.0

    Note
    ----
    Alias for :func:`.float64`.

    Parameters
    ----------
    x : :class:`.NumericExpression` or :class:`.BooleanExpression` or :class:`.StringExpression`

    Returns
    -------
    :class:`.NumericExpression` of type :py:data:`.tfloat64`
    """
    return float64(x)


@typecheck(x=expr_oneof(expr_numeric, expr_bool, expr_str))
def bool(x) -> BooleanExpression:
    """Convert to a Boolean expression.

    Examples
    --------

    >>> hl.eval(hl.bool('TRUE'))
    True

    >>> hl.eval(hl.bool(1.5))
    True

    Notes
    -----
    Numeric expressions return ``True`` if they are non-zero, and ``False``
    if they are zero.

    Acceptable string values are: ``'True'``, ``'true'``, ``'TRUE'``,
    ``'False'``, ``'false'``, and ``'FALSE'``.

    Parameters
    ----------
    x : :class:`.NumericExpression` or :class:`.BooleanExpression` or :class:`.StringExpression`

    Returns
    -------
    :class:`.BooleanExpression`
    """
    if x.dtype == tbool:
        return x
    elif is_numeric(x.dtype):
        return x != 0
    else:
        return x._method("toBoolean", tbool)


@typecheck(contig=expr_str,
           position=expr_int32,
           before=expr_int32,
           after=expr_int32,
           reference_genome=reference_genome_type)
def get_sequence(contig, position, before=0, after=0, reference_genome='default') -> StringExpression:
    """Return the reference sequence at a given locus.

    Examples
    --------

    Return the reference allele for ``'GRCh37'`` at the locus ``'1:45323'``:

    >>> hl.eval(hl.get_sequence('1', 45323, reference_genome='GRCh37')) # doctest: +SKIP
    "T"

    Notes
    -----
    This function requires `reference genome` has an attached
    reference sequence. Use :meth:`.ReferenceGenome.add_sequence` to
    load and attach a reference sequence to a reference genome.

    Returns ``None`` if `contig` and `position` are not valid coordinates in
    `reference_genome`.

    Parameters
    ----------
    contig : :class:`.Expression` of type :py:data:`.tstr`
        Locus contig.
    position : :class:`.Expression` of type :py:data:`.tint32`
        Locus position.
    before : :class:`.Expression` of type :py:data:`.tint32`, optional
        Number of bases to include before the locus of interest. Truncates at
        contig boundary.
    after : :class:`.Expression` of type :py:data:`.tint32`, optional
        Number of bases to include after the locus of interest. Truncates at
        contig boundary.
    reference_genome : :obj:`str` or :class:`.ReferenceGenome`
        Reference genome to use. Must have a reference sequence available.

    Returns
    -------
    :class:`.StringExpression`
    """

    if not reference_genome.has_sequence():
        raise TypeError("Reference genome '{}' does not have a sequence loaded. Use 'add_sequence' to load the sequence from a FASTA file.".format(reference_genome.name))
    return _func("getReferenceSequence({})".format(reference_genome.name), tstr,
                 contig, position, before, after)

@typecheck(contig=expr_str,
           reference_genome=reference_genome_type)
def is_valid_contig(contig, reference_genome='default') -> BooleanExpression:
    """Returns ``True`` if `contig` is a valid contig name in `reference_genome`.

    Examples
    --------

    >>> hl.eval(hl.is_valid_contig('1', reference_genome='GRCh37'))
    True

    >>> hl.eval(hl.is_valid_contig('chr1', reference_genome='GRCh37'))
    False

    Parameters
    ----------
    contig : :class:`.Expression` of type :py:data:`.tstr`
    reference_genome : :obj:`str` or :class:`.ReferenceGenome`

    Returns
    -------
    :class:`.BooleanExpression`
    """
    return _func("isValidContig({})".format(reference_genome.name), tbool, contig)

@typecheck(contig=expr_str,
           position=expr_int32,
           reference_genome=reference_genome_type)
def is_valid_locus(contig, position, reference_genome='default') -> BooleanExpression:
    """Returns ``True`` if `contig` and `position` is a valid site in `reference_genome`.

    Examples
    --------

    >>> hl.eval(hl.is_valid_locus('1', 324254, 'GRCh37'))
    True

    >>> hl.eval(hl.is_valid_locus('chr1', 324254, 'GRCh37'))
    False

    Parameters
    ----------
    contig : :class:`.Expression` of type :py:data:`.tstr`
    position : :class:`.Expression` of type :py:data:`.tint`
    reference_genome : :obj:`str` or :class:`.ReferenceGenome`

    Returns
    -------
    :class:`.BooleanExpression`
    """
    return _func("isValidLocus({})".format(reference_genome.name), tbool, contig, position)

@typecheck(locus=expr_locus(), is_female=expr_bool, father=expr_call, mother=expr_call, child=expr_call)
def mendel_error_code(locus, is_female, father, mother, child):
    r"""Compute a Mendelian violation code for genotypes.

    >>> father = hl.call(0, 0)
    >>> mother = hl.call(1, 1)
    >>> child1 = hl.call(0, 1)  # consistent
    >>> child2 = hl.call(0, 0)  # Mendel error
    >>> locus = hl.locus('2', 2000000)

    >>> hl.eval(hl.mendel_error_code(locus, True, father, mother, child1))
    None

    >>> hl.eval(hl.mendel_error_code(locus, True, father, mother, child2))
    7

    Note
    ----
    Ignores call phasing, and assumes diploid and biallelic. Haploid calls for
    hemiploid samples on sex chromosomes also are acceptable input.

    Notes
    -----
    In the table below, the copy state of a locus with respect to a trio is
    defined as follows, where PAR is the `pseudoautosomal region
    <https://en.wikipedia.org/wiki/Pseudoautosomal_region>`__ (PAR) of X and Y
    defined by the reference genome and the autosome is defined by
    :meth:`.LocusExpression.in_autosome`:

    - Auto -- in autosome or in PAR, or in non-PAR of X and female child
    - HemiX -- in non-PAR of X and male child
    - HemiY -- in non-PAR of Y and male child

    `Any` refers to the set \{ HomRef, Het, HomVar, NoCall \} and `~`
    denotes complement in this set.

    +------+---------+---------+--------+------------+---------------+
    | Code | Dad     | Mom     | Kid    | Copy State | Implicated    |
    +======+=========+=========+========+============+===============+
    |    1 | HomVar  | HomVar  | Het    | Auto       | Dad, Mom, Kid |
    +------+---------+---------+--------+------------+---------------+
    |    2 | HomRef  | HomRef  | Het    | Auto       | Dad, Mom, Kid |
    +------+---------+---------+--------+------------+---------------+
    |    3 | HomRef  | ~HomRef | HomVar | Auto       | Dad, Kid      |
    +------+---------+---------+--------+------------+---------------+
    |    4 | ~HomRef | HomRef  | HomVar | Auto       | Mom, Kid      |
    +------+---------+---------+--------+------------+---------------+
    |    5 | HomRef  | HomRef  | HomVar | Auto       | Kid           |
    +------+---------+---------+--------+------------+---------------+
    |    6 | HomVar  | ~HomVar | HomRef | Auto       | Dad, Kid      |
    +------+---------+---------+--------+------------+---------------+
    |    7 | ~HomVar | HomVar  | HomRef | Auto       | Mom, Kid      |
    +------+---------+---------+--------+------------+---------------+
    |    8 | HomVar  | HomVar  | HomRef | Auto       | Kid           |
    +------+---------+---------+--------+------------+---------------+
    |    9 | Any     | HomVar  | HomRef | HemiX      | Mom, Kid      |
    +------+---------+---------+--------+------------+---------------+
    |   10 | Any     | HomRef  | HomVar | HemiX      | Mom, Kid      |
    +------+---------+---------+--------+------------+---------------+
    |   11 | HomVar  | Any     | HomRef | HemiY      | Dad, Kid      |
    +------+---------+---------+--------+------------+---------------+
    |   12 | HomRef  | Any     | HomVar | HemiY      | Dad, Kid      |
    +------+---------+---------+--------+------------+---------------+


    Parameters
    ----------
    locus : :class:`.LocusExpression`
    is_female : :class:`.BooleanExpression`
    father : :class:`.CallExpression`
    mother : :class:`.CallExpression`
    child : :class:`.CallExpression`

    Returns
    -------
    :class:`.Int32Expression`
    """
    father_n = father.n_alt_alleles()
    mother_n = mother.n_alt_alleles()
    child_n = child.n_alt_alleles()

    auto_cond = (hl.case(missing_false=True)
                 .when((father_n == 2) & (mother_n == 2) & (child_n == 1), 1)
                 .when((father_n == 0) & (mother_n == 0) & (child_n == 1), 2)
                 .when((father_n == 0) & (mother_n == 0) & (child_n == 2), 5)
                 .when((father_n == 2) & (mother_n == 2) & (child_n == 0), 8)
                 .when((father_n == 0) & (child_n == 2), 3)
                 .when((mother_n == 0) & (child_n == 2), 4)
                 .when((father_n == 2) & (child_n == 0), 6)
                 .when((mother_n == 2) & (child_n == 0), 7)
                 .or_missing()
                 )

    hemi_x_cond = (hl.case(missing_false=True)
                   .when((mother_n == 2) & (child_n == 0), 9)
                   .when((mother_n == 0) & (child_n > 0), 10)
                   .or_missing()
                   )

    hemi_y_cond = (hl.case(missing_false=True)
                   .when((father_n > 0) & (child_n == 0), 11)
                   .when((father_n == 0) & (child_n > 0), 12)
                   .or_missing()
                   )

    return (hl.case()
            .when(locus.in_autosome_or_par() | is_female, auto_cond)
            .when(locus.in_x_nonpar() & (~is_female), hemi_x_cond)
            .when(locus.in_y_nonpar() & (~is_female), hemi_y_cond)
            .or_missing()
            )

@typecheck(locus=expr_locus(), alleles=expr_array(expr_str))
def min_rep(locus, alleles):
    """Computes the minimal representation of a (locus, alleles) polymorphism.

    Examples
    --------

    >>> hl.eval(hl.min_rep(hl.locus('1', 100000), ['TAA', 'TA']))
    Struct(locus=Locus(contig=1, position=100000, reference_genome=GRCh37), alleles=['TA', 'T'])

    >>> hl.eval(hl.min_rep(hl.locus('1', 100000), ['AATAA', 'AACAA']))
    Struct(locus=Locus(contig=1, position=100002, reference_genome=GRCh37), alleles=['T', 'C'])

    Notes
    -----
    Computing the minimal representation can cause the locus shift right (the
    position can increase).

    Parameters
    ----------
    locus : :class:`.LocusExpression`
    alleles : :class:`.ArrayExpression` of type :py:data:`.tstr`

    Returns
    -------
    :class:`.StructExpression`
        A :class:`.tstruct` expression with two fields, `locus`
        (:class:`.LocusExpression`) and `alleles`
        (:class:`.ArrayExpression` of type :py:data:`.tstr`).
    """
    ret_type = tstruct(locus=locus.dtype, alleles=alleles.dtype)
    return _func('min_rep', ret_type, locus, alleles)

@typecheck(x=oneof(expr_locus(), expr_interval(expr_locus())),
           dest_reference_genome=reference_genome_type,
           min_match=builtins.float,
           include_strand=builtins.bool)
def liftover(x, dest_reference_genome, min_match=0.95, include_strand=False):
    """Lift over coordinates to a different reference genome.

    Examples
    --------

    Lift over the locus coordinates from reference genome ``'GRCh37'`` to
    ``'GRCh38'``:

    >>> hl.eval(hl.liftover(hl.locus('1', 1034245, 'GRCh37'), 'GRCh38')) # doctest: +SKIP
    Locus(contig='chr1', position=1098865, reference_genome='GRCh38')

    Lift over the locus interval coordinates from reference genome ``'GRCh37'``
    to ``'GRCh38'``:

    >>> hl.eval(hl.liftover(hl.locus_interval('20', 60001, 82456, True, True, 'GRCh37'), 'GRCh38')) # doctest: +SKIP
    Interval(Locus(contig='chr20', position=79360, reference_genome='GRCh38'),
             Locus(contig='chr20', position=101815, reference_genome='GRCh38'),
             True,
             True)

    See :ref:`liftover_howto` for more instructions on lifting over a Table
    or MatrixTable.

    Notes
    -----
    This function requires the reference genome of `x` has a chain file loaded
    for `dest_reference_genome`. Use :meth:`.ReferenceGenome.add_liftover` to
    load and attach a chain file to a reference genome.

    Returns ``None`` if `x` could not be converted.

    Warning
    -------
        Before using the result of :func:`.liftover` as a new row key or column
        key, be sure to filter out missing values.

    Parameters
    ----------
    x : :class:`.Expression` of type :py:data:`.tlocus` or :py:data:`.tinterval` of :py:data:`.tlocus`
        Locus or locus interval to lift over.
    dest_reference_genome : :obj:`str` or :class:`.ReferenceGenome`
        Reference genome to convert to.
    min_match : :obj:`float`
        Minimum ratio of bases that must remap.
    include_strand : :obj:`bool`
        If True, output the result as a :class:`.StructExpression` with the first field `result` being
        the locus or locus interval and the second field `is_negative_strand` is a boolean indicating
        whether the locus or locus interval has been mapped to the negative strand of the destination
        reference genome. Otherwise, output the converted locus or locus interval.

    Returns
    -------
    :class:`.Expression`
        A locus or locus interval converted to `dest_reference_genome`.
    """

    if not 0.0 <= min_match <= 1.0:
        raise TypeError("'liftover' requires 'min_match' is in the range [0, 1]. Got {}".format(min_match))

    if isinstance(x.dtype, tlocus):
        rg = x.dtype.reference_genome
        method_name = "liftoverLocus({})({})".format(rg.name, dest_reference_genome.name)
        rtype = tstruct(result=tlocus(dest_reference_genome), is_negative_strand=tbool)
    else:
        rg = x.dtype.point_type.reference_genome
        method_name = "liftoverLocusInterval({})({})".format(rg.name, dest_reference_genome.name)
        rtype = tstruct(result=tinterval(tlocus(dest_reference_genome)), is_negative_strand=tbool)

    if not rg.has_liftover(dest_reference_genome.name):
        raise TypeError("""Reference genome '{}' does not have liftover to '{}'.
        Use 'add_liftover' to load a liftover chain file.""".format(rg.name, dest_reference_genome.name))

    expr = _func(method_name, rtype, x, to_expr(min_match, tfloat))
    if not include_strand:
        expr = expr.result
    return expr


@typecheck(f=func_spec(1, expr_float64),
           min=expr_float64,
           max=expr_float64)
def uniroot(f: Callable, min, max):
    """Finds a root of the function `f` within the interval `[min, max]`.

    Examples
    --------

    >>> hl.eval(hl.uniroot(lambda x: x - 1, -5, 5))
    1.0

    Notes
    -----
    `f(min)` and `f(max)` must not have the same sign.

    If no root can be found, the result of this call will be `NA` (missing).

    Parameters
    ----------
    f : function ( (arg) -> :class:`.Float64Expression`)
        Must return a :class:`.Float64Expression`.
    min : :class:`.Float64Expression`
    max : :class:`.Float64Expression`

    Returns
    -------
    :class:`.Float64Expression`
        The root of the function `f`.
    """

    new_id = Env.get_uid()
    lambda_result = to_expr(f(construct_variable(new_id, hl.tfloat64)))

    indices, aggregations = unify_all(lambda_result, min, max)
    ir = Uniroot(new_id, lambda_result._ir, min._ir, max._ir)
    return construct_expr(ir, lambda_result._type, indices, aggregations)


@typecheck(f=expr_str, args=expr_any)
def format(f, *args):
    """Returns a formatted string using a specified format string and arguments.

    Examples
    --------

    >>> hl.eval(hl.format('%.3e', 0.09345332))
    '9.345e-02'

    >>> hl.eval(hl.format('%.4f', hl.null(hl.tfloat64)))
    'null'

    >>> hl.eval(hl.format('%s %s %s', 'hello', hl.tuple([3, hl.locus('1', 2453)]), True))
    'hello [3,1:2453] true'

    Notes
    -----
    See the `Java documentation <https://docs.oracle.com/javase/8/docs/api/java/lang/String.html#format-java.lang.String-java.lang.Object...->`__
    for valid format specifiers and arguments.

    Missing values are printed as ``'null'`` except when using the
    format flags `'b'` and `'B'` (printed as ``'false'`` instead).

    Parameters
    ----------
    f : :class:`.StringExpression`
        Java `format string <https://docs.oracle.com/javase/8/docs/api/java/util/Formatter.html#syntax>`__.
    args : variable-length arguments of :class:`.Expression`
        Arguments to format.

    Returns
    -------
    :class:`.StringExpression`
    """

    return _func("format", hl.tstr, f, hl.tuple(args))


@typecheck(x=expr_float64, y=expr_float64, tolerance=expr_float64, absolute=expr_bool, nan_same=expr_bool)
def approx_equal(x, y, tolerance=1e-6, absolute=False, nan_same=False):
    """Tests whether two numbers are approximately equal.

    Examples
    --------
    >>> hl.eval(hl.approx_equal(0.25, 0.2500001))
    True

    >>> hl.eval(hl.approx_equal(0.25, 0.251, tolerance=1e-3, absolute=True))
    False

    Parameters
    ----------
    x : :class:`.NumericExpression`
    y : :class:`.NumericExpression`
    tolerance : :class:`.NumericExpression`
    absolute : :class:`.BooleanExpression`
        If True, compute ``abs(x - y) <= tolerance``. Otherwise, compute
        ``abs(x - y) <= max(tolerance * max(abs(x), abs(y)), 2 ** -1022)``.
    nan_same : :class:`.BooleanExpression`
        If True, then ``NaN == NaN`` will evaluate to True. Otherwise,
        it will return False.

    Returns
    -------
    :class:`.BooleanExpression`
    """

    return _func("approxEqual", hl.tbool, x, y, tolerance, absolute, nan_same)


def _shift_op(x, y, op):
    assert op in ('<<', '>>', '>>>')
    t = x.dtype
    if t == hl.tint64:
        word_size = 64
        zero = hl.int64(0)
    else:
        word_size = 32
        zero = hl.int32(0)

    indices, aggregations = unify_all(x, y)
    return hl.bind(lambda x, y: (
        hl.case()
            .when(y >= word_size, hl.sign(x) if op == '>>' else zero)
            .when(y > 0, construct_expr(ApplyBinaryPrimOp(op, x._ir, y._ir), t, indices, aggregations))
            .or_error('cannot shift by a negative value: ' + hl.str(x) + f" {op} " + hl.str(y))), x, y)

def _bit_op(x, y, op):
    if x.dtype == hl.tint32 and y.dtype == hl.tint32:
        t = hl.tint32
    else:
        t = hl.tint64
    coercer = coercer_from_dtype(t)
    x = coercer.coerce(x)
    y = coercer.coerce(y)

    indices, aggregations = unify_all(x, y)
    return construct_expr(ApplyBinaryPrimOp(op, x._ir, y._ir), t, indices, aggregations)


@typecheck(x=expr_oneof(expr_int32, expr_int64), y=expr_oneof(expr_int32, expr_int64))
def bit_and(x, y):
    """Bitwise and `x` and `y`.

    Examples
    --------
    >>> hl.eval(hl.bit_and(5, 3))
    1

    Notes
    -----
    See `the Python wiki <https://wiki.python.org/moin/BitwiseOperators>`__
    for more information about bit operators.


    Parameters
    ----------
    x : :class:`.Int32Expression` or :class:`.Int64Expression`
    y : :class:`.Int32Expression` or :class:`.Int64Expression`

    Returns
    -------
    :class:`.Int32Expression` or :class:`.Int64Expression`
    """
    return _bit_op(x, y, '&')


@typecheck(x=expr_oneof(expr_int32, expr_int64), y=expr_oneof(expr_int32, expr_int64))
def bit_or(x, y):
    """Bitwise or `x` and `y`.

    Examples
    --------
    >>> hl.eval(hl.bit_or(5, 3))
    7

    Notes
    -----
    See `the Python wiki <https://wiki.python.org/moin/BitwiseOperators>`__
    for more information about bit operators.


    Parameters
    ----------
    x : :class:`.Int32Expression` or :class:`.Int64Expression`
    y : :class:`.Int32Expression` or :class:`.Int64Expression`

    Returns
    -------
    :class:`.Int32Expression` or :class:`.Int64Expression`
    """
    return _bit_op(x, y, '|')


@typecheck(x=expr_oneof(expr_int32, expr_int64), y=expr_oneof(expr_int32, expr_int64))
def bit_xor(x, y):
    """Bitwise exclusive-or `x` and `y`.

    Examples
    --------
    >>> hl.eval(hl.bit_xor(5, 3))
    6

    Notes
    -----
    See `the Python wiki <https://wiki.python.org/moin/BitwiseOperators>`__
    for more information about bit operators.


    Parameters
    ----------
    x : :class:`.Int32Expression` or :class:`.Int64Expression`
    y : :class:`.Int32Expression` or :class:`.Int64Expression`

    Returns
    -------
    :class:`.Int32Expression` or :class:`.Int64Expression`
    """
    return _bit_op(x, y, '^')


@typecheck(x=expr_oneof(expr_int32, expr_int64), y=expr_int32)
def bit_lshift(x, y):
    """Bitwise left-shift `x` by `y`.

    Examples
    --------
    >>> hl.eval(hl.bit_lshift(5, 3))
    40

    >>> hl.eval(hl.bit_lshift(1, 8))
    256

    Unlike Python, Hail integers are fixed-size (32 or 64 bits),
    and bits extended beyond will be ignored:

    >>> hl.eval(hl.bit_lshift(1, 31))
    -2147483648

    >>> hl.eval(hl.bit_lshift(1, 32))
    0

    >>> hl.eval(hl.bit_lshift(hl.int64(1), 32))
    4294967296

    >>> hl.eval(hl.bit_lshift(hl.int64(1), 64))
    0

    Notes
    -----
    See `the Python wiki <https://wiki.python.org/moin/BitwiseOperators>`__
    for more information about bit operators.

    Parameters
    ----------
    x : :class:`.Int32Expression` or :class:`.Int64Expression`
    y : :class:`.Int32Expression` or :class:`.Int64Expression`

    Returns
    -------
    :class:`.Int32Expression` or :class:`.Int64Expression`
    """
    return _shift_op(x, y, '<<')


@typecheck(x=expr_oneof(expr_int32, expr_int64), y=expr_int32, logical=builtins.bool)
def bit_rshift(x, y, logical=False):
    """Bitwise right-shift `x` by `y`.

    Examples
    --------
    >>> hl.eval(hl.bit_rshift(256, 3))
    32

    With ``logical=False`` (default), the sign is preserved:

    >>> hl.eval(hl.bit_rshift(-1, 1))
    -1

    With ``logical=True``, the sign bit is treated as any other:

    >>> hl.eval(hl.bit_rshift(-1, 1, logical=True))
    2147483647

    Notes
    -----
    If `logical` is ``False``, then the shift is a sign-preserving right shift.
    If `logical` is ``True``, then the shift is logical, with the sign bit
    treated as any other bit.

    See `the Python wiki <https://wiki.python.org/moin/BitwiseOperators>`__
    for more information about bit operators.

    Parameters
    ----------
    x : :class:`.Int32Expression` or :class:`.Int64Expression`
    y : :class:`.Int32Expression` or :class:`.Int64Expression`
    logical : :obj:`bool`

    Returns
    -------
    :class:`.Int32Expression` or :class:`.Int64Expression`
    """
    if logical:
        return _shift_op(x, y, '>>>')
    else:
        return _shift_op(x, y, '>>')


@typecheck(x=expr_oneof(expr_int32, expr_int64))
def bit_not(x):
    """Bitwise invert `x`.

    Examples
    --------
    >>> hl.eval(hl.bit_not(0))
    -1

    Notes
    -----
    See `the Python wiki <https://wiki.python.org/moin/BitwiseOperators>`__
    for more information about bit operators.


    Parameters
    ----------
    x : :class:`.Int32Expression` or :class:`.Int64Expression`

    Returns
    -------
    :class:`.Int32Expression` or :class:`.Int64Expression`
    """
    return construct_expr(ApplyUnaryPrimOp('~', x._ir), x.dtype, x._indices, x._aggregations)


@typecheck(s=expr_str)
def _escape_string(s):
    return _func("escapeString", hl.tstr, s)

@typecheck(l=expr_any, r=expr_any, tolerance=expr_float64, absolute=expr_bool)
def _values_similar(l, r, tolerance=1e-6, absolute=False):
    assert l.dtype == r.dtype
    return ((is_missing(l) & is_missing(r))
            | ((is_defined(l) & is_defined(r)) & _func("valuesSimilar", hl.tbool, l, r, tolerance, absolute)))


@typecheck(coords=expr_array(expr_array(expr_float64)), radius=expr_float64)
def _locus_windows_per_contig(coords, radius):
    rt = hl.ttuple(hl.tarray(hl.tint32), hl.tarray(hl.tint32))
    return _func("locus_windows_per_contig", rt, coords, radius)
