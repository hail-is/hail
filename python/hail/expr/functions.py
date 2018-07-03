import builtins
import math

import hail
import hail as hl
from hail.expr.expr_ast import *
from hail.expr.expressions import *
from hail.expr.expressions.expression_typecheck import *
from hail.expr.types import *
from hail.genetics.reference_genome import reference_genome_type, ReferenceGenome
from hail.typecheck import *
from hail.utils import LinkedList

Coll_T = TypeVar('Collection_T', ArrayExpression, SetExpression)
Num_T = TypeVar('Numeric_T', Int32Expression, Int64Expression, Float32Expression, Float64Expression)


def _func(name, ret_type, *args):
    indices, aggregations = unify_all(*args)
    return construct_expr(ApplyMethod(name, *(a._ast for a in args)), ret_type, indices, aggregations)


@typecheck(t=hail_type)
def null(t: Union[HailType, str]):
    """Creates an expression representing a missing value of a specified type.

    Examples
    --------

    >>> hl.null(hl.tarray(hl.tstr)).value
    None

    >>> hl.null('array<str>').value
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
    return construct_expr(Literal('NA: {}'.format(t._jtype.parsableString())), t)


@typecheck(x=anytype, dtype=nullable(hail_type))
def literal(x: Any, dtype: Optional[Union[HailType, str]] = None):
    """Captures and broadcasts a Python variable or object as an expression.

    Examples
    --------

    >>> table = hl.utils.range_table(8)
    >>> greetings = hl.literal({1: 'Good morning', 4: 'Good afternoon', 6 : 'Good evening'})
    >>> table.annotate(greeting = greetings.get(table.idx)).show()
    +-------+----------------+
    | index | greeting       |
    +-------+----------------+
    | int32 | str            |
    +-------+----------------+
    |     0 | NA             |
    |     1 | Good morning   |
    |     2 | NA             |
    |     3 | NA             |
    |     4 | Good afternoon |
    |     5 | NA             |
    |     6 | Good evening   |
    |     7 | NA             |
    +-------+----------------+

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
    if dtype:
        try:
            dtype._typecheck(x)
        except TypeError as e:
            raise TypeError("'literal': object did not match the passed type '{}'"
                            .format(dtype)) from e
    else:
        dtype = impute_type(x)

    def get_float_str(x):
        if math.isnan(x):
            return 'nan'
        elif math.isinf(x):
            return 'inf' if x > 0 else 'neginf'
        else:
            return builtins.str(builtins.float(x))

    if x is None:
        return hl.null(dtype)
    elif is_primitive(dtype):
        if dtype == tint32:
            assert isinstance(x, builtins.int)
            assert tint32.min_value <= x <= tint32.max_value
            return construct_expr(Literal('i32#{}'.format(x)), tint32)
        elif dtype == tint64:
            assert isinstance(x, builtins.int)
            assert tint64.min_value <= x <= tint64.max_value
            return construct_expr(Literal('i64#{}'.format(x)), tint64)
        elif dtype == tfloat32:
            assert isinstance(x, (builtins.float, builtins.int))
            return construct_expr(Literal('f32#{}'.format(get_float_str(x))), tfloat32)
        elif dtype == tfloat64:
            assert isinstance(x, (builtins.float, builtins.int))
            return construct_expr(Literal('f64#{}'.format(get_float_str(x))), tfloat64)
        elif dtype == tbool:
            assert isinstance(x, builtins.bool)
            return construct_expr(Literal('true' if x else 'false'), tbool)
        else:
            assert dtype == tstr
            assert isinstance(x, builtins.str)
            return construct_expr(Literal('"{}"'.format(escape_str(x))), tstr)
    else:
        return construct_expr(Broadcast(x, dtype), dtype)

@typecheck(condition=expr_bool, consequent=expr_any, alternate=expr_any, missing_false=bool)
def cond(condition,
         consequent,
         alternate,
         missing_false: bool = False):
    """Expression for an if/else statement; tests a condition and returns one of two options based on the result.

    Examples
    --------

    >>> x = 5
    >>> hl.cond(x < 2, 'Hi', 'Bye').value
    'Bye'

    >>> a = hl.literal([1, 2, 3, 4])
    >>> hl.cond(hl.len(a) > 0, 2.0 * a, a / 2.0).value
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

    return construct_expr(Condition(condition._ast, consequent._ast, alternate._ast),
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
    >>> expr.value
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
    >>> expr.value
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

    Expressions are "inlined", leading to perhaps unexpected behavior
    when randomness is involved. For example, let us define a variable
    `x` from the :meth:`.rand_unif` method:

    >>> x = hl.rand_unif(0, 1)

    Note that evaluating `x` multiple times returns different results.
    The value of evaluating `x` is unknown when the expression is defined.

    >>> x.value
    0.3189309481038456

    >>> x.value
    0.20842918568366375

    What if we evaluate `x` multiple times?

    >>> hl.array([x, x, x]).value
    [0.49582541026815163, 0.8549329234134524, 0.7016124997911775]

    The random number generator is called separately for each inclusion
    of `x`. This method, :func:`.bind`, is the solution to this problem!

    >>> hl.bind(lambda y: [y, y, y], x).value
    [0.7897028763765286, 0.7897028763765286, 0.7897028763765286]

    :func:`.bind` also can take multiple arguments:

    >>> hl.bind(lambda x, y: x / y, x, x).value
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
    asts = []

    for expr in exprs:
        uid = Env.get_uid()
        args.append(construct_expr(VariableReference(uid), expr._type, expr._indices, expr._aggregations))
        uids.append(uid)
        asts.append(expr._ast)


    lambda_result = to_expr(f(*args))
    indices, aggregations = unify_all(*exprs, lambda_result)
    ast = Bind(uids, asts, lambda_result._ast)
    return construct_expr(ast, lambda_result.dtype, indices, aggregations)


@typecheck(c1=expr_int32, c2=expr_int32, c3=expr_int32, c4=expr_int32)
def chisq(c1, c2, c3, c4) -> StructExpression:
    """Performs chi-squared test of independence on a 2x2 contingency table.

    Examples
    --------

    >>> hl.chisq(10, 10, 10, 10).value
    Struct(p_value=1.0, odds_ratio=1.0)

    >>> hl.chisq(51, 43, 22, 92).value
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
    return _func("chisq", ret_type, c1, c2, c3, c4)


@typecheck(c1=expr_int32, c2=expr_int32, c3=expr_int32, c4=expr_int32, min_cell_count=expr_int32)
def ctt(c1, c2, c3, c4, min_cell_count) -> StructExpression:
    """Performs chi-squared or Fisher's exact test of independence on a 2x2
    contingency table.

    Examples
    --------

    >>> hl.ctt(51, 43, 22, 92, min_cell_count=22).value
    Struct(p_value=1.4626257805267089e-07, odds_ratio=4.959830866807611)

    >>> hl.ctt(51, 43, 22, 92, min_cell_count=23).value
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
    return _func("ctt", ret_type, c1, c2, c3, c4, min_cell_count)


@typecheck(collection=expr_oneof(expr_dict(),
                                 expr_set(expr_tuple([expr_any, expr_any])),
                                 expr_array(expr_tuple([expr_any, expr_any]))))
def dict(collection) -> DictExpression:
    """Creates a dictionary.

    Examples
    --------

    >>> hl.dict([('foo', 1), ('bar', 2), ('baz', 3)]).value
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

    >> hl.dbeta(.2, 5, 20).value
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

    >>> hl.dpois(5, 3).value
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

    >>> hl.exp(2).value
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

    >>> hl.fisher_exact_test(10, 10, 10, 10).value
    Struct(p_value=1.0000000000000002, odds_ratio=1.0,
           ci_95_lower=0.24385796914260355, ci_95_upper=4.100747675033819)

    >>> hl.fisher_exact_test(51, 43, 22, 92).value
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
    return _func("fet", ret_type, c1, c2, c3, c4)


@typecheck(x=expr_oneof(expr_float32, expr_float64))
def floor(x):
    """The largest integral value that is less than or equal to `x`.

    Examples
    --------

    >>> hl.floor(3.1).value
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

    >>> hl.ceil(3.1).value
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
def hardy_weinberg_p(n_hom_ref, n_het, n_hom_var) -> StructExpression:
    """Performs test of Hardy-Weinberg equilibrium.

    Examples
    --------

    >>> hl.hardy_weinberg_p(250, 500, 250).value
    Struct(r_expected_het_freq=0.5002501250625313, p_hwe=0.9747844394217698)

    >>> hl.hardy_weinberg_p(37, 200, 85).value
    Struct(r_expected_het_freq=0.48964964307448583, p_hwe=1.1337210383168987e-06)

    Notes
    -----
    This method performs a two-sided exact test with mid-p-value correction of
    `Hardy-Weinberg equilibrium <https://en.wikipedia.org/wiki/Hardy%E2%80%93Weinberg_principle>`__
    via an efficient implementation of the
    `Levene-Haldane distribution <https://hail.is/docs/devel/LeveneHaldane.pdf>`__,
    which models the number of heterozygous individuals under equilibrium.

    The mean of this distribution is ``(n_hom_ref * n_hom_var) / (2n - 1)`` where
    ``n = n_hom_ref + n_het + n_hom_var``. So the expected frequency of heterozygotes
    under equilibrium, `r_expected_het_freq`, is this mean divided by ``n``.

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
        A struct expression with two fields, `r_expected_het_freq`
        (:py:data:`.tfloat64`) and `p_value` (:py:data:`.tfloat64`).
    """
    ret_type = tstruct(r_expected_het_freq=tfloat64,
                       p_hwe=tfloat64)
    return _func("hwe", ret_type, n_hom_ref, n_het, n_hom_var)


@typecheck(structs=expr_array(expr_struct()),
           identifier=str)
def index(structs, identifier):
    if not isinstance(structs.dtype.element_type, tstruct):
        raise TypeError("'index' expects an array with element type 'Struct', found '{}'"
                        .format(structs.dtype))
    struct_type = structs.dtype.element_type

    if identifier not in struct_type:
        raise RuntimeError("`structs' does not have a field with identifier `{}'. " \
                           "Struct type is {}.".format(identifier, struct_type))

    key_type = struct_type[identifier]
    value_type = tstruct(**{f: t for f, t in struct_type.items() if f != identifier})

    ast = StructOp('index', structs._ast, identifier)
    return construct_expr(ast, tdict(key_type, value_type),
                          structs._indices, structs._aggregations)


@typecheck(contig=expr_str, pos=expr_int32,
           reference_genome=reference_genome_type)
def locus(contig, pos, reference_genome: Union[str, ReferenceGenome] = 'default') -> LocusExpression:
    """Construct a locus expression from a chromosome and position.

    Examples
    --------

    >>> hl.locus("1", 10000).value
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
    indices, aggregations = unify_all(contig, pos)
    return construct_expr(ApplyMethod('Locus({})'.format(reference_genome.name), contig._ast, pos._ast),
                          tlocus(reference_genome), indices, aggregations)


@typecheck(global_pos=expr_int64,
           reference_genome=reference_genome_type)
def locus_from_global_position(global_pos,
                               reference_genome: Union[str, ReferenceGenome] = 'default') -> LocusExpression:
    """Constructs a locus expression from a global position and a reference genome.
    The inverse of :meth:`.LocusExpression.global_position`.

    Examples
    --------
    >>> hl.locus_from_global_position(0).value
    Locus(contig=1, position=1, reference_genome=GRCh37)

    >>> hl.locus_from_global_position(2824183054).value
    Locus(contig=21, position=42584230, reference_genome=GRCh37)

    >>> hl.locus_from_global_position(2824183054, 'GRCh38').value
    Locus(contig=22, position=1, reference_genome=GRCh38)

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
    return construct_expr(ApplyMethod('globalPosToLocus({})'.format(reference_genome.name), global_pos._ast),
                          tlocus(reference_genome), global_pos._indices, global_pos._aggregations)


@typecheck(s=expr_str,
           reference_genome=reference_genome_type)
def parse_locus(s, reference_genome: Union[str, ReferenceGenome] = 'default') -> LocusExpression:
    """Construct a locus expression by parsing a string or string expression.

    Examples
    --------

    >>> hl.parse_locus("1:10000").value
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
    return construct_expr(ApplyMethod('Locus({})'.format(reference_genome.name), s._ast), tlocus(reference_genome),
                          s._indices, s._aggregations)


@typecheck(s=expr_str,
           reference_genome=reference_genome_type)
def parse_variant(s, reference_genome: Union[str, ReferenceGenome] = 'default') -> StructExpression:
    """Construct a struct with a locus and alleles by parsing a string.

    Examples
    --------

    >>> hl.parse_variant('1:100000:A:T,C').value
    Struct(locus=Locus('1', 100000), alleles=['A', 'T', 'C'])

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
    return construct_expr(ApplyMethod('LocusAlleles({})'.format(reference_genome.name), s._ast), t,
                          s._indices, s._aggregations)


@typecheck(gp=expr_array(expr_float64))
def gp_dosage(gp) -> Float64Expression:
    """
    Return expected genotype dosage from array of genotype probabilities.

    Examples
    --------

    >>> hl.gp_dosage([0.0, 0.5, 0.5]).value
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

    >>> hl.pl_dosage([5, 10, 100]).value
    0.24025307377482674

    Parameters
    ----------
    pl : :class:`.ArrayNumericExpression` of type :py:data:`.tint32`
        Length 3 array of bi-allelic Phred-scaled genotype likelihoods

    Returns
    -------
    :class:`.Expression` of type :py:data:`.tfloat64`
    """
    return _func("plDosage", tfloat64, pl)


@typecheck(start=expr_any, end=expr_any,
           includes_start=expr_bool, includes_end=expr_bool)
def interval(start,
             end,
             includes_start=True,
             includes_end=False) -> IntervalExpression:
    """Construct an interval expression.

    Examples
    --------

    >>> hl.interval(5, 100).value
    Interval(start=5, end=100)

    >>> hl.interval(hl.locus("1", 100), hl.locus("1", 1000)).value
    Interval(start=Locus(contig=1, position=100, reference_genome=GRCh37),
             end=Locus(contig=1, position=1000, reference_genome=GRCh37))

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

    indices, aggregations = unify_all(start, end, includes_start, includes_end)

    return construct_expr(
        ApplyMethod('Interval', start._ast, end._ast, includes_start._ast, includes_end._ast), tinterval(start.dtype),
        indices, aggregations)


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

    >>> hl.locus_interval("1", 100, 1000).value
    Interval(start=Locus(contig=1, position=100, reference_genome=GRCh37),
             end=Locus(contig=1, position=1000, reference_genome=GRCh37))

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
    indices, aggregations = unify_all(contig, start, end, includes_start, includes_end)

    return construct_expr(
        ApplyMethod('LocusInterval({})'.format(reference_genome.name),
                    contig._ast, start._ast, end._ast, includes_start._ast, includes_end._ast),
        tinterval(tlocus(reference_genome)), indices, aggregations)


@typecheck(s=expr_str,
           reference_genome=reference_genome_type)
def parse_locus_interval(s, reference_genome: Union[str, ReferenceGenome] = 'default') -> IntervalExpression:
    """Construct a locus interval expression by parsing a string or string
    expression.

    Examples
    --------

    >>> hl.parse_locus_interval('1:1000-2000').value
    Interval(start=Locus(contig=1, position=1000, reference_genome=GRCh37),
             end=Locus(contig=1, position=2000, reference_genome=GRCh37))

    >>> hl.parse_locus_interval('1:start-10M').value
    Interval(start=Locus(contig=1, position=0, reference_genome=GRCh37),
             end=Locus(contig=1, position=10000000, reference_genome=GRCh37))

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
    return construct_expr(
        ApplyMethod('LocusInterval({})'.format(reference_genome.name), s._ast),
        tinterval(tlocus(reference_genome)),
        s._indices, s._aggregations)


@typecheck(alleles=expr_int32,
           phased=expr_bool)
def call(*alleles, phased=False) -> CallExpression:
    """Construct a call expression.

    Examples
    --------

    >>> hl.call(1, 0).value
    Call(alleles=[1, 0], phased=False)

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
    indices, aggregations = unify_all(phased, *alleles)
    if builtins.len(alleles) > 2:
        raise NotImplementedError("'call' supports a maximum of 2 alleles.")
    return construct_expr(ApplyMethod('Call', *[a._ast for a in alleles], phased._ast), tcall, indices, aggregations)


@typecheck(gt_index=expr_int32)
def unphased_diploid_gt_index_call(gt_index) -> CallExpression:
    """Construct an unphased, diploid call from a genotype index.

    Examples
    --------

    >>> hl.unphased_diploid_gt_index_call(4).value
    Call(alleles=[1, 2], phased=False)

    Parameters
    ----------
    gt_index : :obj:`int` or :class:`.Expression` of type :py:data:`.tint32`
        Unphased, diploid genotype index.

    Returns
    -------
    :class:`.CallExpression`
    """
    gt_index = to_expr(gt_index)
    return construct_expr(ApplyMethod('UnphasedDiploidGtIndexCall', gt_index._ast), tcall, gt_index._indices,
                          gt_index._aggregations)


@typecheck(s=expr_str)
def parse_call(s) -> CallExpression:
    """Construct a call expression by parsing a string or string expression.

    Examples
    --------

    >>> hl.parse_call('0|2').value
    Call([0, 2], phased=True)

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
    s = to_expr(s)
    return construct_expr(ApplyMethod('Call', s._ast), tcall, s._indices, s._aggregations)


@typecheck(expression=expr_any)
def is_defined(expression) -> BooleanExpression:
    """Returns ``True`` if the argument is not missing.

    Examples
    --------

    >>> hl.is_defined(5).value
    True

    >>> hl.is_defined(hl.null(hl.tstr)).value
    False

    >>> hl.is_defined(hl.null(hl.tbool) & True).value
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
    return _func("isDefined", tbool, expression)


@typecheck(expression=expr_any)
def is_missing(expression) -> BooleanExpression:
    """Returns ``True`` if the argument is missing.

    Examples
    --------

    >>> hl.is_missing(5).value
    False

    >>> hl.is_missing(hl.null(hl.tstr)).value
    True

    >>> hl.is_missing(hl.null(hl.tbool) & True).value
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
    return _func("isMissing", tbool, expression)


@typecheck(x=expr_oneof(expr_float32, expr_float64))
def is_nan(x) -> BooleanExpression:
    """Returns ``True`` if the argument is ``nan`` (not a number).

    Examples
    --------

    >>> hl.is_nan(0).value
    False

    >>> hl.is_nan(hl.literal(0) / 0).value
    True

    >>> hl.is_nan(hl.literal(0) / hl.null(hl.tfloat64)).value
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


@typecheck(x=expr_any)
def json(x) -> StringExpression:
    """Convert an expression to a JSON string expression.

    Examples
    --------

    >>> hl.json([1,2,3,4,5]).value
    '[1,2,3,4,5]'

    >>> hl.json(hl.struct(a='Hello', b=0.12345, c=[1,2], d={'hi', 'bye'})).value
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

    >>> hl.log(10).value
    2.302585092994046

    >>> hl.log(10, 10).value
    1.0

    >>> hl.log(1024, 2).value
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

    >>> hl.log10(1000).value
    3.0

    >>> hl.log10(0.0001123).value
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
    >>> hl.coalesce(x1, x2).value
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
    def make_case(*expr_args):
        c = case()
        for e in expr_args:
            c = c.when(hl.is_defined(e), e)
        return c.or_missing()
    return bind(make_case, *exprs)

@typecheck(a=expr_any, b=expr_any)
def or_else(a, b):
    """If `a` is missing, return `b`.

    Examples
    --------

    >>> hl.or_else(5, 7).value
    5

    >>> hl.or_else(hl.null(hl.tint32), 7).value
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
    return hl.cond(hl.is_defined(a), a, b)

@typecheck(predicate=expr_bool, value=expr_any)
def or_missing(predicate, value):
    """Returns `value` if `predicate` is ``True``, otherwise returns missing.

    Examples
    --------

    >>> hl.or_missing(True, 5).value
    5

    >>> hl.or_missing(False, 5).value
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

    >>> hl.binom_test(5, 10, 0.5, 'less').value
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

    >>> hl.pchisqtail(5, 1).value
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

    >>> hl.pnorm(0).value
    0.5

    >>> hl.pnorm(1).value
    0.8413447460685429

    >>> hl.pnorm(2).value
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
    """The cumulative probability function of a Poisson distribution.

    Examples
    --------

    >>> hl.ppois(2, 1).value
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

    >>> hl.qchisqtail(0.01, 1).value
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

    >>> hl.qnorm(0.90).value
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
    """Inverts :meth:`.ppois`.

    Examples
    --------

    >>> hl.qpois(0.99, 1).value
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

    >>> hl.range(0, 10).value
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

    >>> hl.range(0, 10, step=3).value
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
    return _func("range", tarray(tint32), start, stop, step)


@typecheck(p=expr_float64)
def rand_bool(p) -> BooleanExpression:
    """Returns ``True`` with probability `p` (RNG).

    Examples
    --------

    >>> hl.rand_bool(0.5).value
    True

    >>> hl.rand_bool(0.5).value
    False

    Warning
    -------
    This function is non-deterministic, meaning that successive runs of the same pipeline including
    RNG expressions may return different results. This is a known bug.

    Parameters
    ----------
    p : float or :class:`.Expression` of type :py:data:`.tfloat64`
        Probability between 0 and 1.

    Returns
    -------
    :class:`.BooleanExpression`
    """
    return _func("pcoin", tbool, p)


@typecheck(mean=expr_float64, sd=expr_float64)
def rand_norm(mean=0, sd=1) -> Float64Expression:
    """Samples from a normal distribution with mean `mean` and standard deviation `sd` (RNG).

    Examples
    --------

    >>> hl.rand_norm().value
    1.5388475315213386

    >>> hl.rand_norm().value
    -0.3006188509144124

    Warning
    -------
    This function is non-deterministic, meaning that successive runs of the same
    pipeline including RNG expressions may return different results. This is a known
    bug.

    Parameters
    ----------
    mean : float or :class:`.Expression` of type :py:data:`.tfloat64`
        Mean of normal distribution.
    sd : float or :class:`.Expression` of type :py:data:`.tfloat64`
        Standard deviation of normal distribution.

    Returns
    -------
    :class:`.Expression` of type :py:data:`.tfloat64`
    """
    return _func("rnorm", tfloat64, mean, sd)


@typecheck(lamb=expr_float64)
def rand_pois(lamb) -> Float64Expression:
    """Samples from a Poisson distribution with rate parameter `lamb` (RNG).

    Examples
    --------

    >>> hl.rand_pois(1).value
    2.0

    >>> hl.rand_pois(1).value
    3.0

    Warning
    -------
    This function is non-deterministic, meaning that successive runs of the same
    pipeline including RNG expressions may return different results. This is a known
    bug.

    Parameters
    ----------
    lamb : float or :class:`.Expression` of type :py:data:`.tfloat64`
        Rate parameter for Poisson distribution.

    Returns
    -------
    :class:`.Expression` of type :py:data:`.tfloat64`
    """
    return _func("rpois", tfloat64, lamb)


@typecheck(min=expr_float64, max=expr_float64)
def rand_unif(min, max) -> Float64Expression:
    """Returns a random floating-point number uniformly drawn from the interval [`min`, `max`].

    Examples
    --------

    >>> hl.rand_unif(0, 1).value
    0.7983073825816226

    >>> hl.rand_unif(0, 1).value
    0.5161799497741769

    Warning
    -------
    This function is non-deterministic, meaning that successive runs of the same
    pipeline including RNG expressions may return different results. This is a known
    bug.

    Parameters
    ----------
    min : float or :class:`.Expression` of type :py:data:`.tfloat64`
        Left boundary of range.
    max : float or :class:`.Expression` of type :py:data:`.tfloat64`
        Right boundary of range.

    Returns
    -------
    :class:`.Expression` of type :py:data:`.tfloat64`
    """
    return _func("runif", tfloat64, min, max)


@typecheck(x=expr_float64)
def sqrt(x) -> Float64Expression:
    """Returns the square root of `x`.

    Examples
    --------

    >>> hl.sqrt(3).value
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

    >>> hl.is_snp('A', 'T').value
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

    >>> hl.is_mnp('AA', 'GT').value
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

    >>> hl.is_transition('A', 'T').value
    False

    >>> hl.is_transition('AAA', 'AGA').value
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

    >>> hl.is_transversion('A', 'T').value
    True

    >>> hl.is_transversion('AAA', 'AGA').value
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

    >>> hl.is_insertion('A', 'ATT').value
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

    >>> hl.is_deletion('ATT', 'A').value
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

    >>> hl.is_indel('ATT', 'A').value
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

    >>> hl.is_deletion('A', '*').value
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

    >>> hl.is_deletion('ATT', 'GCA').value
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

    >>> hl.is_strand_ambiguous('A', 'T').value
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

    >>> hl.allele_type('A', 'T').value
    'SNP'

    >>> hl.allele_type('ATT', 'A').value
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

    >>> hl.hamming('ATATA', 'ATGCA').value
    2

    >>> hl.hamming('abcdefg', 'zzcdefz').value
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

    >>> hl.entropy('ac').value
    1.0

    >>> hl.entropy('accctg').value
    1.79248

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

    >>> hl.str(hl.struct(a=5, b=7)).value
    '{"a": 5, "b": 7}'

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

    >>> hl.downcode(hl.call(1, 2), 2).value
    Call(alleles=[0, 2], phased=False)

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

    >>> hl.gq_from_pl([0, 69, 1035]).value
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

    >>> hl.triangle(3).value
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

    >>> hl.filter(lambda x: x % 2 == 0, a).value
    [2, 4]

    >>> hl.filter(lambda x: ~(x[-1] == 'e'), s).value
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
    return collection._bin_lambda_method("filter", f, collection.dtype.element_type, lambda _: collection.dtype)


@typecheck(f=func_spec(1, expr_bool),
           collection=expr_oneof(expr_set(), expr_array()))
def any(f: Callable, collection) -> BooleanExpression:
    """Returns ``True`` if `f` returns ``True`` for any element.

    Examples
    --------

    >>> a = ['The', 'quick', 'brown', 'fox']
    >>> s = {1, 3, 5, 6, 7, 9}

    >>> hl.any(lambda x: x[-1] == 'x', a).value
    True

    >>> hl.any(lambda x: x % 4 == 0, s).value
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

    return collection._bin_lambda_method("exists", f, collection.dtype.element_type, lambda _: tbool)


@typecheck(f=func_spec(1, expr_bool),
           collection=expr_oneof(expr_set(), expr_array()))
def all(f: Callable, collection) -> BooleanExpression:
    """Returns ``True`` if `f` returns ``True`` for every element.

    Examples
    --------

    >>> a = ['The', 'quick', 'brown', 'fox']
    >>> s = {1, 3, 5, 6, 7, 9}

    >>> hl.all(lambda x: hl.len(x) > 3, a).value
    False

    >>> hl.all(lambda x: x < 10, s).value
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

    return collection._bin_lambda_method("forall", f, collection.dtype.element_type, lambda _: tbool)


@typecheck(f=func_spec(1, expr_bool),
           collection=expr_oneof(expr_set(), expr_array()))
def find(f: Callable, collection):
    """Returns the first element where `f` returns ``True``.

    Examples
    --------

    >>> a = ['The', 'quick', 'brown', 'fox']
    >>> s = {1, 3, 5, 6, 7, 9}

    >>> hl.find(lambda x: x[-1] == 'x', a).value
    'fox'

    >>> hl.find(lambda x: x % 4 == 0, s).value
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

    >>> hl.flatmap(lambda x: x[1:], a).value
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

    return collection._bin_lambda_method("flatMap", f, collection.dtype.element_type, unify_ret)


@typecheck(f=func_spec(1, expr_any),
           collection=expr_oneof(expr_set(), expr_array()))
def group_by(f: Callable, collection) -> DictExpression:
    """Group collection elements into a dict according to a lambda function.

    Examples
    --------

    >>> a = ['The', 'quick', 'brown', 'fox']

    >>> hl.group_by(lambda x: hl.len(x), a).value
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
    return collection._bin_lambda_method("groupBy", f,
                                         collection.dtype.element_type,
                                         lambda t: tdict(t, collection.dtype))


@typecheck(arrays=expr_array(), fill_missing=bool)
def zip(*arrays, fill_missing: bool = False) -> ArrayExpression:
    """Zip together arrays into a single array.

    Examples
    --------

    >>> hl.zip([1], [10, 20], [100, 200, 300]).value
    [(1, 10, 100)]

    >>> hl.zip([1], [10, 20], [100, 200, 300], fill_missing=True).value
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

@typecheck(a=expr_array())
def zip_with_index(a):
    """Returns an array of (index, element) tuples.

    Examples
    --------

    >>> hl.zip_with_index(['A', 'B', 'C']).value
    [(0, 'A'), (1, 'B'), (2, 'C')]

    Parameters
    ----------
    a : :class:`.ArrayExpression`

    Returns
    -------
    :class:`.ArrayExpression`
        Array of (index, element) tuples.
    """
    return bind(lambda aa: range(0, len(aa)).map(lambda i: (i, aa[i])), a)

@typecheck(f=func_spec(1, expr_any),
           collection=expr_oneof(expr_set(), expr_array()))
def map(f: Callable, collection):
    """Transform each element of a collection.

    Examples
    --------

    >>> a = ['The', 'quick', 'brown', 'fox']

    >>> hl.map(lambda x: hl.len(x), a).value
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
    return collection._bin_lambda_method("map", f,
                                         collection.dtype.element_type,
                                         lambda t: collection.dtype.__class__(t))


@typecheck(x=expr_oneof(expr_set(), expr_array(), expr_dict(), expr_str, expr_tuple(), expr_struct()))
def len(x) -> Int32Expression:
    """Returns the size of a collection or string.

    Examples
    --------

    >>> a = ['The', 'quick', 'brown', 'fox']
    >>> s = {1, 3, 5, 6, 7, 9}

    >>> hl.len(a).value
    4

    >>> hl.len(s).value
    6

    >>> hl.len("12345").value
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
    else:
        return x._method("size", tint32)


@typecheck(exprs=expr_oneof(expr_numeric, expr_set(expr_numeric), expr_array(expr_numeric)))
def max(*exprs) -> NumericExpression:
    """Returns the maximum element of a collection or of given numeric expressions.

    Examples
    --------

    Take the maximum value of an array:

    >>> hl.max([1, 3, 5, 6, 7, 9]).value
    9

    Take the maximum value of values:

    >>> hl.max(1, 50, 2).value
    50

    Notes
    -----
    Like the Python builtin ``max`` function, this function can either take a
    single iterable expression (an array or set of numeric elements), or
    variable-length arguments of numeric expressions.

    Parameters
    ----------
    exprs : :class:`.ArrayExpression` or :class:`.SetExpression` or varargs of :class:`.NumericExpression`
        Single numeric array or set, or multiple numeric values.

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
        return expr._method('max', expr.dtype.element_type)
    else:
        if not builtins.all(is_numeric(e.dtype) for e in exprs):
            raise TypeError("'max' expects a single numeric array expression or multiple numeric expressions\n"
                            "  Found {} arguments with types '{}'".format(builtins.len(exprs), ', '.join(
                "'{}'".format(e.dtype) for e in exprs)))
        ret_t = unify_types(*(e.dtype for e in exprs))
        exprs = tuple(e._promote_numeric(ret_t) for e in exprs)
        if builtins.len(exprs) == 2:
            return exprs[0]._method('max', ret_t, exprs[1])
        else:
            return max([e for e in exprs])


@typecheck(exprs=expr_oneof(expr_numeric, expr_set(expr_numeric), expr_array(expr_numeric)))
def min(*exprs) -> NumericExpression:
    """Returns the minimum of a collection or of given numeric expressions.

    Examples
    --------

    Take the minimum value of an array:

    >>> hl.min([2, 3, 5, 6, 7, 9]).value
    2

    Take the minimum value:

    >>> hl.min(12, 50, 2).value
    2

    Notes
    -----
    Like the Python builtin ``min`` function, this function can either take a
    single iterable expression (an array or set of numeric elements), or
    variable-length arguments of numeric expressions.

    Parameters
    ----------
    exprs : :class:`.ArrayExpression` or :class:`.SetExpression` or varargs of :class:`.NumericExpression`
        Single numeric array or set, or multiple numeric values.

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
        return expr._method('min', expr.dtype.element_type)
    else:
        if not builtins.all(is_numeric(e.dtype) for e in exprs):
            raise TypeError("'min' expects a single numeric array expression or multiple numeric expressions\n"
                            "  Found {} arguments with types '{}'".format(builtins.len(exprs), ', '.join(
                "'{}'".format(e.dtype) for e in exprs)))
        ret_t = unify_types(*(e.dtype for e in exprs))
        exprs = tuple(e._promote_numeric(ret_t) for e in exprs)
        if builtins.len(exprs) == 2:
            return exprs[0]._method('min', ret_t, exprs[1])
        else:
            return min([e for e in exprs])


@typecheck(x=expr_oneof(expr_numeric, expr_array(expr_numeric)))
def abs(x):
    """Take the absolute value of a numeric value or array.

    Examples
    --------

    >>> hl.abs(-5).value
    5

    >>> hl.abs([1.0, -2.5, -5.1]).value
    [1.0, 2.5, 5.1]

    Parameters
    ----------
    x : :class:`.NumericExpression` or :class:`.ArrayNumericExpression`

    Returns
    -------
    :class:`.NumericExpression` or :class:`.ArrayNumericExpression`.
    """
    if isinstance(x.dtype, tarray):
        return map(abs, x)
    else:
        return x._method('abs', x.dtype)


@typecheck(x=expr_oneof(expr_numeric, expr_array(expr_numeric)))
def sign(x):
    """Returns the sign of a numeric value or array.

    Examples
    --------

    >>> hl.sign(-1.23).value
    -1.0

    >>> hl.sign([-4, 0, 5]).value
    [-1, 0, 1]

    >>> hl.sign([0.0, 3.14]).value
    [0.0, 1.0]

    >>> hl.sign(float('nan')).value  # doctest: +SKIP
    nan

    Notes
    -----
    The sign function preserves type and maps ``nan`` to ``nan``.

    Parameters
    ----------
    x : :class:`.NumericExpression` or :class:`.ArrayNumericExpression`

    Returns
    -------
    :class:`.NumericExpression` or :class:`.ArrayNumericExpression`.
    """
    if isinstance(x.dtype, tarray):
        return map(sign, x)
    else:
        return x._method('sign', x.dtype)


@typecheck(collection=expr_oneof(expr_set(expr_numeric), expr_array(expr_numeric)))
def mean(collection) -> Float64Expression:
    """Returns the mean of all values in the collection.

    Examples
    --------

    >>> a = [1, 3, 5, 6, 7, 9]

    >>> hl.mean(a).value
    5.2

    Note
    ----
    Missing elements are ignored.

    Parameters
    ----------
    collection : :class:`.ArrayExpression` or :class:`.SetExpression`
        Collection expression with numeric element type.

    Returns
    -------
    :class:`.Expression` of type :py:data:`.tfloat64`
    """
    return collection._method("mean", tfloat64)


@typecheck(collection=expr_oneof(expr_set(expr_numeric), expr_array(expr_numeric)))
def median(collection) -> NumericExpression:
    """Returns the median value in the collection.

    Examples
    --------

    >>> a = [1, 3, 5, 6, 7, 9]

    >>> hl.median(a).value
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


@typecheck(collection=expr_oneof(expr_set(expr_numeric), expr_array(expr_numeric)))
def product(collection) -> NumericExpression:
    """Returns the product of values in the collection.

    Examples
    --------

    >>> a = [1, 3, 5, 6, 7, 9]

    >>> hl.product(a).value
    5670

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
    return collection._method("product", collection.dtype.element_type)


@typecheck(collection=expr_oneof(expr_set(expr_numeric), expr_array(expr_numeric)))
def sum(collection) -> NumericExpression:
    """Returns the sum of values in the collection.

    Examples
    --------

    >>> a = [1, 3, 5, 6, 7, 9]

    >>> hl.sum(a).value
    31

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
    return collection._method("sum", collection.dtype.element_type)


@typecheck(kwargs=expr_any)
def struct(**kwargs) -> StructExpression:
    """Construct a struct expression.

    Examples
    --------

    >>> s = hl.struct(a=5, b='Foo')
    >>> s.a.value
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
    >>> t.value
    (1, 2, '3')

    >>> t[2].value
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
    >>> s.show()
    {'Alice', 'Bob', 'Charlie'}

    Returns
    -------
    :class:`.SetExpression`
        Set of all unique elements.
    """
    if isinstance(collection.dtype, tset):
        return collection
    return collection._method("toSet", tset(collection.dtype.element_type))


@typecheck(t=hail_type)
def empty_set(t: Union[HailType, str]) -> SetExpression:
    """Returns an empty set of elements of a type `t`.

    Examples
    --------

    >>> hl.empty_set(hl.tstr).value
    set()

    Parameters
    ----------
    t : :obj:`str` or :class:`.HailType`
        Type of the set elements.

    Returns
    -------
    :class:`.SetExpression`
    """
    return filter(lambda x: False, set([null(t)]))


@typecheck(collection=expr_oneof(expr_set(), expr_array(), expr_dict()))
def array(collection) -> ArrayExpression:
    """Construct an array expression.

    Examples
    --------

    >>> s = {'Bob', 'Charlie', 'Alice'}

    >>> hl.array(s).value
    ['Charlie', 'Alice', 'Bob']

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
        return collection._method("toArray", tarray(collection.dtype.element_type))
    else:
        assert isinstance(collection.dtype, tdict)
        return _func('dictToArray', tarray(ttuple(collection.dtype.key_type, collection.dtype.value_type)), collection)


@typecheck(t=hail_type)
def empty_array(t: Union[HailType, str]) -> ArrayExpression:
    """Returns an empty array of elements of a type `t`.

    Examples
    --------

    >>> hl.empty_array(hl.tint32).value
    []

    Parameters
    ----------
    t : :obj:`str` or :class:`.HailType`
        Type of the array elements.

    Returns
    -------
    :class:`.ArrayExpression`
    """
    return filter(lambda x: False, array([null(t)]))


@typecheck(key_type=hail_type, value_type=hail_type)
def empty_dict(key_type: Union[HailType, str], value_type: Union[HailType, str]) -> DictExpression:
    """Returns an empty dictionary with key type `key_type` and value type
    `value_type`.

    Examples
    --------

    >>> hl.empty_dict(hl.tstr, hl.tint32).value
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

    >>> hl.flatten(a).value
    [1, 2, 2, 3]

    Parameters
    ----------
    collection : :class:`.ArrayExpression` or :class:`.SetExpression`
        Collection with element type :class:`.tarray` or :class:`.tset`.

    Returns
    -------
    collection : :class:`.ArrayExpression` or :class:`.SetExpression`
    """
    return collection._method("flatten", collection._type.element_type)


@typecheck(collection=expr_oneof(expr_array(), expr_set()),
           delimiter=expr_str)
def delimit(collection, delimiter=',') -> StringExpression:
    """Joins elements of `collection` into single string delimited by `delimiter`.

    Examples
    --------

    >>> a = ['Bob', 'Charlie', 'Alice', 'Bob', 'Bob']

    >>> hl.delimit(a).value
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

    >>> hl.sorted(a).value
    ['Alice', 'Bob', 'Charlie']

    >>> hl.sorted(a, reverse=False).value
    ['Charlie', 'Bob', 'Alice']

    >>> hl.sorted(a, key=lambda x: hl.len(x)).value
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
    ascending = ~reverse

    def can_sort(t):
        return t == tstr or is_numeric(t)

    if key is None:
        if not can_sort(collection.dtype.element_type):
            raise TypeError("'sorted' expects an array with element type 'String' or numeric, found '{}'"
                            .format(collection.dtype))
        return collection._method("sort", collection.dtype, ascending)
    else:
        def check_f(t):
            if not can_sort(t):
                raise TypeError("'sort_by' expects 'key' to return type 'String' or numeric, found '{}'".format(t))
            return collection.dtype

        return collection._bin_lambda_method("sortBy", key, collection.dtype.element_type, check_f, ascending)


@typecheck(array=expr_array(expr_numeric), unique=bool)
def argmin(array, unique: bool = False) -> Int32Expression:
    """Return the index of the minimum value in the array.

    Examples
    --------

    >>> hl.argmin([0.2, 0.3, 0.6]).value
    0

    >>> hl.argmin([0.4, 0.2, 0.2]).value
    1

    >>> hl.argmin([0.4, 0.2, 0.2], unique=True).value
    None

    Notes
    -----
    Returns the index of the minimum value in the array.

    If two or more elements are tied for minimum, then the `unique` parameter
    will determine the result. If `unique` is ``False``, then the first index
    will be returned. If `unique` is ``True``, then the result is missing.

    If the array is empty, then the result is missing.

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

    >>> hl.argmax([0.2, 0.2, 0.6]).value
    2

    >>> hl.argmax([0.4, 0.4, 0.2]).value
    0

    >>> hl.argmax([0.4, 0.4, 0.2], unique=True).value
    None

    Notes
    -----
    Returns the index of the maximum value in the array.

    If two or more elements are tied for maximum, then the `unique` parameter
    will determine the result. If `unique` is ``False``, then the first index
    will be returned. If `unique` is ``True``, then the result is missing.

    If the array is empty, then the result is missing.

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

    >>> hl.float64('1.1').value
    1.1

    >>> hl.float64(1).value
    1.0

    >>> hl.float64(True).value
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

    >>> hl.float32('1.1').value
    1.1

    >>> hl.float32(1).value
    1.0

    >>> hl.float32(True).value
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

    >>> hl.int64('1').value
    1

    >>> hl.int64(1.5).value
    1

    >>> hl.int64(True).value
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

    >>> hl.int32('1').value
    1

    >>> hl.int32(1.5).value
    1

    >>> hl.int32(True).value
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

    >>> hl.int('1').value
    1

    >>> hl.int(1.5).value
    1

    >>> hl.int(True).value
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

    >>> hl.float('1.1').value
    1.1

    >>> hl.float(1).value
    1.0

    >>> hl.float(True).value
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

    >>> hl.bool('TRUE').value
    True

    >>> hl.bool(1.5).value
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

    >>> hl.get_sequence('1', 45323, 'GRCh37').value # doctest: +SKIP
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

    >>> hl.is_valid_contig('1', 'GRCh37').value
    True

    >>> hl.is_valid_contig('chr1', 'GRCh37').value
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

    >>> hl.is_valid_locus('1', 324254, 'GRCh37').value
    True

    >>> hl.is_valid_locus('chr1', 324254, 'GRCh37').value
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
    """Compute a Mendelian violation code for genotypes.

    >>> father = hl.call(0, 0)
    >>> mother = hl.call(1, 1)
    >>> child1 = hl.call(0, 1)  # consistent
    >>> child2 = hl.call(0, 0)  # Mendel error
    >>> locus = hl.locus('2', 2000000)

    >>> hl.mendel_error_code(locus, True, father, mother, child1).value
    None

    >>> hl.mendel_error_code(locus, True, father, mother, child2).value
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

    >>> hl.min_rep(hl.locus('1', 100000), ['TAA', 'TA']).value
    (Locus(contig=1, position=100000, reference_genome=GRCh37), ['TA', 'T'])

    >>> hl.min_rep(hl.locus('1', 100000), ['AATAA', 'AACAA']).value
    (Locus(contig=1, position=100002, reference_genome=GRCh37), ['T', 'C'])

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
    :class:`.TupleExpression`
        Tuple of (:class:`.LocusExpression`, :class:`.ArrayExpression` of type :py:data:`.tstr`)
    """
    return _func('min_rep', hl.ttuple(locus.dtype, alleles.dtype), locus, alleles)

@typecheck(x=oneof(expr_locus(), expr_interval(expr_locus())),
           dest_reference_genome=reference_genome_type,
           min_match=builtins.float)
def liftover(x, dest_reference_genome, min_match=0.95):
    """Lift over coordinates to a different reference genome.

    Examples
    --------

    Lift over the locus coordinates from reference genome ``'GRCh37'`` to
    ``'GRCh38'``:

    >>> hl.liftover(hl.locus('1', 1034245, 'GRCh37'), 'GRCh38').value # doctest: +SKIP
    Locus(contig='chr1', position=1098865, reference_genome='GRCh38')

    Lift over the locus interval coordinates from reference genome ``'GRCh37'``
    to ``'GRCh38'``:

    >>> hl.liftover(hl.locus_interval('20', 60001, 82456, True, True, 'GRCh37'), 'GRCh38').value # doctest: +SKIP
    Interval(Locus(contig='chr20', position=79360, reference_genome='GRCh38'),
             Locus(contig='chr20', position=101815, reference_genome='GRCh38'),
             True,
             True)

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
    min_match : :class:`.Expression` of type :py:data:`.tfloat64`
        Minimum ratio of bases that must remap.

    Returns
    -------
    :class:`.Expression`
        A locus or locus interval converted to `dest_reference_genome`.
    """

    if not 0.0 <= min_match <= 1.0:
        raise TypeError("'liftover' requires 'min_match' is in the range [0, 1]. Got {}".format(min_match))

    if isinstance(x.dtype, tlocus):
        rg = x.dtype.reference_genome
        method_name = "liftoverLocus({})".format(rg.name)
        rtype = tlocus(dest_reference_genome)
    else:
        rg = x.dtype.point_type.reference_genome
        method_name = "liftoverLocusInterval({})".format(rg.name)
        rtype = tinterval(tlocus(dest_reference_genome))

    if not rg.has_liftover(dest_reference_genome.name):
        raise TypeError("""Reference genome '{}' does not have liftover to '{}'.
        Use 'add_liftover' to load a liftover chain file.""".format(rg.name, dest_reference_genome.name))

    return _func(method_name, rtype, to_expr(dest_reference_genome.name, tstr), x, to_expr(min_match, tfloat))


@typecheck(f=func_spec(1, expr_float64),
           min=expr_float64,
           max=expr_float64)
def uniroot(f: Callable, min, max):
    """Finds a root of the function `f` within the interval `[min, max]`.

    Examples
    --------

    >>> hl.uniroot(lambda x: x - 1, -5, 5).value
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
    lambda_result = to_expr(f(construct_expr(VariableReference(new_id), hl.tfloat64)))

    indices, aggregations = unify_all(lambda_result, min, max)
    ast = LambdaFunction("uniroot", new_id, lambda_result._ast, min._ast, max._ast)
    return hl.expr.expressions.construct_expr(ast, lambda_result._type, indices, aggregations)
