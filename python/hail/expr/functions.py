import builtins

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
    indices, aggregations, joins = unify_all(*args)
    return construct_expr(ApplyMethod(name, *(a._ast for a in args)), ret_type, indices, aggregations, joins)


@typecheck(t=hail_type)
def null(t: Union[HailType, str]) -> Expression:
    """Creates an expression representing a missing value of a specified type.

    Examples
    --------
    .. doctest::

        >>> hl.eval_expr(hl.null(hl.tarray(hl.tstr)))
        None

        >>> hl.eval_expr(hl.null('array<str>'))
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
def literal(x: Any, dtype: Optional[Union[HailType, str]] = None) -> Expression:
    """Captures and broadcasts a Python variable or object as an expression.

    Examples
    --------
    .. doctest::

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
            return construct_expr(Literal('f32#{}'.format(builtins.float(x))), tfloat32)
        elif dtype == tfloat64:
            assert isinstance(x, (builtins.float, builtins.int))
            return construct_expr(Literal('f64#{}'.format(builtins.float(x))), tfloat64)
        elif dtype == tbool:
            assert isinstance(x, builtins.bool)
            return construct_expr(Literal('true' if x else 'false'), tbool)
        else:
            assert dtype == tstr
            assert isinstance(x, builtins.str)
            return construct_expr(Literal('"{}"'.format(escape_str(x))), tstr)
    else:
        uid = Env.get_uid()

        def joiner(obj):
            json = dtype._to_json(x)
            if isinstance(obj, hl.Table):
                return hl.Table(obj._jt.annotateGlobalJSON(json, dtype._jtype, uid))
            else:
                return hl.MatrixTable(obj._jvds.annotateGlobalJSON(json, dtype._jtype, uid))

        return construct_expr(Select(TopLevelReference('global', Indices()), uid),
                              dtype, joins=LinkedList(Join).push(Join(joiner, [uid], uid, [])))


@typecheck(condition=expr_bool, consequent=expr_any, alternate=expr_any, missing_false=bool)
def cond(condition: BooleanExpression,
         consequent: Expression,
         alternate: Expression,
         *,
         missing_false: bool = False) -> Expression:
    """Expression for an if/else statement; tests a condition and returns one of two options based on the result.

    Examples
    --------
    .. doctest::

        >>> x = 5
        >>> hl.eval_expr( hl.cond(x < 2, 'Hi', 'Bye') )
        'Bye'

        >>> a = hl.literal([1, 2, 3, 4])
        >>> hl.eval_expr( hl.cond(hl.len(a) > 0,
        ...                   2.0 * a,
        ...                   a / 2.0) )
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
        condition = hl.bind(condition, lambda x: hl.is_defined(x) & x)
    indices, aggregations, joins = unify_all(condition, consequent, alternate)
    t = unify_types_limited(consequent.dtype, alternate.dtype)
    if t is None:
        raise TypeError("'cond' requires the 'consequent' and 'alternate' arguments to have the same type\n"
                        "    consequent: type {}\n"
                        "    alternate:  type {}".format(consequent._type, alternate._type))
    return construct_expr(Condition(condition._ast, consequent._ast, alternate._ast),
                          t, indices, aggregations, joins)


def case(missing_false: bool=False) -> 'hail.expr.builders.CaseBuilder':
    """Chain multiple if-else statements with a :class:`.CaseBuilder`.

    Examples
    --------
    .. doctest::

        >>> x = hl.literal('foo bar baz')
        >>> expr = (hl.case()
        ...                  .when(x[:3] == 'FOO', 1)
        ...                  .when(hl.len(x) == 11, 2)
        ...                  .when(x == 'secret phrase', 3)
        ...                  .default(0))
        >>> hl.eval_expr(expr)
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
def switch(expr: Expression) -> 'hail.expr.builders.SwitchBuilder':
    """Build a conditional tree on the value of an expression.

    Examples
    --------
    .. doctest::

        >>> csq = hl.literal('loss of function')
        >>> expr = (hl.switch(csq)
        ...                  .when('synonymous', 1)
        ...                  .when('SYN', 1)
        ...                  .when('missense', 2)
        ...                  .when('MIS', 2)
        ...                  .when('loss of function', 3)
        ...                  .when('LOF', 3)
        ...                  .or_missing())
        >>> hl.eval_expr(expr)
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


@typecheck(expr=expr_any, f=func_spec(1, expr_any))
def bind(expr: Expression, f: Callable[[Expression], Expression]) -> Expression:
    """Bind a temporary variable and use it in a function.

    Examples
    --------
    Expressions are "inlined", leading to perhaps unexpected behavior
    when randomness is involved. For example, let us define a variable
    `x` from the :meth:`.rand_unif` method:

    >>> x = hl.rand_unif(0, 1)

    Note that evaluating `x` multiple times returns different results.
    The value of evaluating `x` is unknown when the expression is defined.

    .. doctest::

        >>> hl.eval_expr(x)
        0.3189309481038456

        >>> hl.eval_expr(x)
        0.20842918568366375

    What if we evaluate `x` multiple times in the same invocation of
    :meth:`~hail.expr.eval_expr`?

    .. doctest::

        >>> hl.eval_expr([x, x, x])
        [0.49582541026815163, 0.8549329234134524, 0.7016124997911775]

    The random number generator is called separately for each inclusion
    of `x`. This method, `bind`, is the solution to this problem!

    .. doctest::

        >>> hl.eval_expr(hl.bind(x, lambda y: [y, y, y]))
        [0.7897028763765286, 0.7897028763765286, 0.7897028763765286]

    Parameters
    ----------
    expr : :class:`.Expression`
        Expression to bind.
    f : function ( (arg) -> :class:`.Expression`)
        Function of `expr`.

    Returns
    -------
    :class:`.Expression`
        Result of evaluating `f` with `expr` as an argument.
    """
    uid = Env.get_uid()
    expr = to_expr(expr)

    f_input = construct_expr(VariableReference(uid), expr._type, expr._indices, expr._aggregations, expr._joins)
    lambda_result = to_expr(f(f_input))

    indices, aggregations, joins = unify_all(expr, lambda_result)
    ast = Bind(uid, expr._ast, lambda_result._ast)
    return construct_expr(ast, lambda_result._type, indices, aggregations, joins)


@typecheck(c1=expr_int32, c2=expr_int32, c3=expr_int32, c4=expr_int32)
def chisq(c1: Int32Expression, c2: Int32Expression, c3: Int32Expression, c4: Int32Expression) -> Float64Expression:
    """Calculates p-value (Chi-square approximation) and odds ratio for a 2x2 table.

    Examples
    --------
    .. doctest::

        >>> hl.eval_expr(hl.chisq(10, 10,
        ...                   10, 10))
        Struct(odds_ratio=1.0, p_value=1.0)

        >>> hl.eval_expr(hl.chisq(30, 30,
        ...                   50, 10))
        Struct(odds_ratio=0.2, p_value=0.000107511176729)

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
def ctt(c1: Int32Expression,
        c2: Int32Expression,
        c3: Int32Expression,
        c4: Int32Expression,
        min_cell_count: Int32Expression) -> Float64Expression:
    """Calculates p-value and odds ratio for 2x2 table.

    Examples
    --------
    .. doctest::

        >>> hl.eval_expr(hl.ctt(10, 10,
        ...              10, 10, min_cell_count=15))
        Struct(odds_ratio=1.0, p_value=1.0)

        >>> hl.eval_expr(hl.ctt(30, 30,
        ...              50, 10, min_cell_count=15))
        Struct(odds_ratio=0.202874620964, p_value=0.000190499944324)

    Notes
    -----
     If any cell is lower than `min_cell_count`, Fisher's exact test is used. Otherwise, faster
     chi-squared approximation is used.

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
        Minimum cell count for chi-squared approximation.

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
def dict(collection: Union[Mapping, DictExpression, SetExpression, ArrayExpression]) -> DictExpression:
    """Creates a dictionary.

    Examples
    --------
    .. doctest::

        >>> hl.eval_expr(hl.dict([('foo', 1), ('bar', 2), ('baz', 3)]))
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
def dbeta(x: Float64Expression, a: Float64Expression, b: Float64Expression) -> Float64Expression:
    """
    Returns the probability density at `x` of a `beta distribution
    <https://en.wikipedia.org/wiki/Beta_distribution>`__ with parameters `a`
    (alpha) and `b` (beta).

    Examples
    --------
    .. doctest::

        >> hl.eval_expr(hl.dbeta(.2, 5, 20))
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
    """
    return _func("dbeta", tfloat64, x, a, b)


@typecheck(x=expr_float64, lamb=expr_float64, log_p=expr_bool)
def dpois(x: Float64Expression, lamb: Float64Expression, log_p: BooleanExpression = False) -> Float64Expression:
    """Compute the (log) probability density at x of a Poisson distribution with rate parameter `lamb`.

    Examples
    --------
    .. doctest::

        >>> hl.eval_expr(hl.dpois(5, 3))
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
def exp(x: Float64Expression) -> Float64Expression:
    """Computes `e` raised to the power `x`.

    Examples
    --------
    .. doctest::

        >>> hl.eval_expr(hl.exp(2))
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
def fisher_exact_test(c1: Int32Expression,
                      c2: Int32Expression,
                      c3: Int32Expression,
                      c4: Int32Expression) -> Float64Expression:
    """Calculates the p-value, odds ratio, and 95% confidence interval with Fisher's exact test for a 2x2 table.

    Examples
    --------
    .. doctest::

        >>> hl.eval_expr(hl.fisher_exact_test(10, 10,
        ...                                   10, 10))
        Struct(p_value=1.0000000000000002, odds_ratio=1.0,
               ci_95_lower=0.24385796914260355, ci_95_upper=4.100747675033819)

        >>> hl.eval_expr(hl.fisher_exact_test(30, 30,
        ...                                   50, 10))
        Struct(p_value=0.00019049994432397886, odds_ratio=0.20287462096407916,
               ci_95_lower=0.07687933053900567, ci_95_upper=0.4987032678214519)

    Notes
    -----
    This method is identical to the version implemented in
    `R <https://stat.ethz.ch/R-manual/R-devel/library/stats/html/fisher.test.html>`_ with default
    parameters (two-sided, alpha = 0.05, null hypothesis that the odds ratio equals 1).

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
def floor(x: NumericExpression) -> NumericExpression:
    """The largest integral value that is less than or equal to `x`.

    Examples
    --------
    .. doctest::

        >>> hl.eval_expr(hl.floor(3.1))
        3.0

    Returns
    -------
    :obj:`.Float64Expression`
    """
    return _func("floor", x.dtype, x)


@typecheck(x=expr_oneof(expr_float32, expr_float64))
def ceil(x: NumericExpression) -> NumericExpression:
    """The smallest integral value that is greater than or equal to `x`.

    Examples
    --------
    .. doctest::

        >>> hl.eval_expr(hl.ceil(3.1))
        4.0

    Returns
    -------
    :obj:`.Float64Expression`
    """
    return _func("ceil", x.dtype, x)


@typecheck(n_hom_ref=expr_int32, n_het=expr_int32, n_hom_var=expr_int32)
def hardy_weinberg_p(n_hom_ref: Int32Expression,
                     n_het: Int32Expression,
                     n_hom_var: Int32Expression) -> Float64Expression:
    """Compute Hardy-Weinberg Equilbrium p-value and heterozygosity ratio.

    Examples
    --------
    .. doctest::

        >>> hl.eval_expr(hl.hardy_weinberg_p(20, 50, 26))
        Struct(r_expected_het_freq=0.500654450262, p_hwe=0.762089599352)

        >>> hl.eval_expr(hl.hardy_weinberg_p(37, 200, 85))
        Struct(r_expected_het_freq=0.489649643074, p_hwe=1.13372103832e-06)

    Notes
    -----
    For more information, see the
    `Wikipedia page <https://en.wikipedia.org/wiki/Hardy%E2%80%93Weinberg_principle>`__

    Parameters
    ----------
    n_hom_ref : int or :class:`.Expression` of type :py:data:`.tint32`
        Homozygous reference count.
    n_het : int or :class:`.Expression` of type :py:data:`.tint32`
        Heterozygote count.
    n_hom_var : int or :class:`.Expression` of type :py:data:`.tint32`
        Homozygous alternate count.

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
                          structs._indices, structs._aggregations, structs._joins)


@typecheck(contig=expr_str, pos=expr_int32,
           reference_genome=reference_genome_type)
def locus(contig: StringExpression, pos: Int32Expression,
          reference_genome: Union[str, ReferenceGenome] = 'default') -> LocusExpression:
    """Construct a locus expression from a chromosome and position.

    Examples
    --------
    .. doctest::

        >>> hl.eval_expr(hl.locus("1", 10000))
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
    indices, aggregations, joins = unify_all(contig, pos)
    return construct_expr(ApplyMethod('Locus({})'.format(reference_genome.name), contig._ast, pos._ast),
                          tlocus(reference_genome), indices, aggregations, joins)


@typecheck(s=expr_str,
           reference_genome=reference_genome_type)
def parse_locus(s: StringExpression, reference_genome: Union[str, ReferenceGenome] = 'default'):
    """Construct a locus expression by parsing a string or string expression.

    Examples
    --------
    .. doctest::

        >>> hl.eval_expr(hl.parse_locus("1:10000"))
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
                          s._indices, s._aggregations, s._joins)


@typecheck(s=expr_str,
           reference_genome=reference_genome_type)
def parse_variant(s: StringExpression, reference_genome: Union[str, ReferenceGenome] = 'default') -> StructExpression:
    """Construct a struct with a locus and alleles by parsing a string.

    Examples
    --------
    .. doctest::

        >>> hl.eval_expr(hl.parse_variant('1:100000:A:T,C'))
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
                          s._indices, s._aggregations, s._joins)


@typecheck(gp=expr_array(expr_float64))
def gp_dosage(gp: ArrayNumericExpression) -> Float64Expression:
    """
    Return expected genotype dosage from array of genotype probabilities.

    Examples
    --------
    .. doctest::

        >>> hl.eval_expr(hl.gp_dosage([0.0, 0.5, 0.5]))
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
def pl_dosage(pl):
    """
    Return expected genotype dosage from array of Phred-scaled genotype
    likelihoods with uniform prior. Only defined for bi-allelic variants. The
    `pl` argument must be length 3.

    Examples
    --------
    .. doctest::

        >>> hl.eval_expr(hl.pl_dosage([5, 10, 100]))
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
def interval(start: Expression,
             end: Expression,
             includes_start: BooleanExpression = True,
             includes_end: BooleanExpression = False) -> IntervalExpression:
    """Construct an interval expression.

    Examples
    --------
    .. doctest::

        >>> hl.eval_expr(hl.interval(5, 100))
        Interval(start=5, end=100)

        >>> hl.eval_expr(hl.interval(hl.locus("1", 100),
        ...                          hl.locus("1", 1000)))
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

    indices, aggregations, joins = unify_all(start, end, includes_start, includes_end)

    return construct_expr(
        ApplyMethod('Interval', start._ast, end._ast, includes_start._ast, includes_end._ast), tinterval(start.dtype),
        indices, aggregations, joins)


@typecheck(contig=expr_str, start=expr_int32,
           end=expr_int32, includes_start=expr_bool,
           includes_end=expr_bool, reference_genome=reference_genome_type)
def locus_interval(contig: StringExpression,
                   start: Int32Expression,
                   end: Int32Expression,
                   includes_start: BooleanExpression = True,
                   includes_end: BooleanExpression = False,
                   reference_genome: Union[str, ReferenceGenome] = 'default') -> IntervalExpression:
    """Construct a locus interval expression.

    Examples
    --------
    .. doctest::

        >>> hl.eval_expr(hl.locus_interval("1", 100, 1000))
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
    indices, aggregations, joins = unify_all(contig, start, end, includes_start, includes_end)

    return construct_expr(
        ApplyMethod('LocusInterval({})'.format(reference_genome.name),
                    contig._ast, start._ast, end._ast, includes_start._ast, includes_end._ast),
        tinterval(tlocus(reference_genome)), indices, aggregations, joins)


@typecheck(s=expr_str,
           reference_genome=reference_genome_type)
def parse_locus_interval(s: StringExpression,
                         reference_genome: Union[str, ReferenceGenome] = 'default') -> IntervalExpression:
    """Construct a locus interval expression by parsing a string or string
    expression.

    Examples
    --------
    .. doctest::

        >>> hl.eval_expr(hl.parse_locus_interval('1:1000-2000'))
        Interval(start=Locus(contig=1, position=1000, reference_genome=GRCh37),
                 end=Locus(contig=1, position=2000, reference_genome=GRCh37))

        >>> hl.eval_expr(hl.parse_locus_interval('1:start-10M'))
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
        s._indices, s._aggregations, s._joins)


@typecheck(alleles=expr_int32,
           phased=expr_bool)
def call(*alleles: Int32Expression, phased: BooleanExpression = False) -> CallExpression:
    """Construct a call expression.

    Examples
    --------
    .. doctest::

        >>> hl.eval_expr(hl.call(1, 0))
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
    indices, aggregations, joins = unify_all(phased, *alleles)
    if builtins.len(alleles) > 2:
        raise NotImplementedError("'call' supports a maximum of 2 alleles.")
    return construct_expr(ApplyMethod('Call', *[a._ast for a in alleles], phased._ast), tcall, indices, aggregations,
                          joins)


@typecheck(gt_index=expr_int32)
def unphased_diploid_gt_index_call(gt_index: Int32Expression) -> CallExpression:
    """Construct an unphased, diploid call from a genotype index.

    Examples
    --------
    .. doctest::

        >>> hl.eval_expr(hl.unphased_diploid_gt_index_call(4))
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
                          gt_index._aggregations, gt_index._joins)


@typecheck(s=expr_str)
def parse_call(s: StringExpression) -> CallExpression:
    """Construct a call expression by parsing a string or string expression.

    Examples
    --------
    .. doctest::

        >>> hl.eval_expr(hl.parse_call('0|2'))
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
    return construct_expr(ApplyMethod('Call', s._ast), tcall, s._indices, s._aggregations, s._joins)


@typecheck(expression=expr_any)
def is_defined(expression: Expression) -> BooleanExpression:
    """Returns ``True`` if the argument is not missing.

    Examples
    --------
    .. doctest::

        >>> hl.eval_expr(hl.is_defined(5))
        True

        >>> hl.eval_expr(hl.is_defined(hl.null(hl.tstr)))
        False

        >>> hl.eval_expr(hl.is_defined(hl.null(hl.tbool) & True))
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
def is_missing(expression: Expression) -> BooleanExpression:
    """Returns ``True`` if the argument is missing.

    Examples
    --------
    .. doctest::

        >>> hl.eval_expr(hl.is_missing(5))
        False

        >>> hl.eval_expr(hl.is_missing(hl.null(hl.tstr)))
        True

        >>> hl.eval_expr(hl.is_missing(hl.null(hl.tbool) & True))
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
def is_nan(x: NumericExpression) -> BooleanExpression:
    """Returns ``True`` if the argument is ``NaN`` (not a number).

    Examples
    --------
    .. doctest::

        >>> hl.eval_expr(hl.is_nan(0))
        False

        >>> hl.eval_expr(hl.is_nan(hl.literal(0) / 0))
        True

        >>> hl.eval_expr(hl.is_nan(hl.literal(0) / hl.null(hl.tfloat64)))
        None

    Notes
    -----
    Note that :meth:`.is_missing` will return ``False`` on ``NaN`` since ``NaN``
    is a defined value. Additionally, this method will return missing if `x` is
    missing.

    Parameters
    ----------
    x : float or :class:`.Expression` of type :py:data:`.tfloat64`
        Expression to test.

    Returns
    -------
    :class:`.BooleanExpression`
        ``True`` if `x` is ``NaN``, ``False`` otherwise.
    """
    return _func("isnan", tbool, x)


@typecheck(x=expr_any)
def json(x: Expression) -> StringExpression:
    """Convert an expression to a JSON string expression.

    Examples
    --------
    .. doctest::

        >>> hl.eval_expr(hl.json([1,2,3,4,5]))
        '[1,2,3,4,5]'

        >>> hl.eval_expr(hl.json(hl.struct(a='Hello', b=0.12345, c=[1,2], d={'hi', 'bye'})))
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
def log(x: Float64Expression, base: Optional[Float64Expression] = None) -> Float64Expression:
    """Take the logarithm of the `x` with base `base`.

    Examples
    --------
    .. doctest::

        >>> hl.eval_expr(hl.log(10))
        2.302585092994046

        >>> hl.eval_expr(hl.log(10, 10))
        1.0

        >>> hl.eval_expr(hl.log(1024, 2))
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
def log10(x: Float64Expression) -> Float64Expression:
    """Take the logarithm of the `x` with base 10.

    Examples
    --------
    .. doctest::

        >>> hl.eval_expr(hl.log10(1000))
        3.0

        >>> hl.eval_expr(hl.log10(0.0001123))
        -3.949620243738542

    Parameters
    ----------
    x : float or :class:`.Expression` of type :py:data:`.tfloat64`

    Returns
    -------
    :class:`.Expression` of type :py:data:`.tfloat64`
    """
    return _func("log10", tfloat64, x)


@typecheck(a=expr_any, b=expr_any)
def or_else(a: Expression, b: Expression) -> Expression:
    """If `a` is missing, return `b`.

    Examples
    --------
    .. doctest::

        >>> hl.eval_expr(hl.or_else(5, 7))
        5

        >>> hl.eval_expr(hl.or_else(hl.null(hl.tint32), 7))
        7

    Parameters
    ----------
    a
    b

    Returns
    -------
    :class:`.Expression`
    """
    t = unify_types(a._type, b._type)
    if t is None:
        raise TypeError("'or_else' requires the 'a' and 'b' arguments to have the same type\n"
                        "    a: type {}\n"
                        "    b:  type {}".format(a._type, b._type))
    return _func("orElse", t, a, b)


@typecheck(predicate=expr_bool, value=expr_any)
def or_missing(predicate: BooleanExpression, value: Expression):
    """Returns `value` if `predicate` is ``True``, otherwise returns missing.

    Examples
    --------
    .. doctest::

        >>> hl.eval_expr(hl.or_missing(True, 5))
        5

        >>> hl.eval_expr(hl.or_missing(False, 5))
        None

    Parameters
    ----------
    predicate : bool or :class:`.BooleanExpression`
    value : Value to return if `predicate` is true.

    Returns
    -------
    :class:`.Expression`
        This expression has the same type as `b`.
    """
    predicate = to_expr(predicate)
    return _func("orMissing", value._type, predicate, value)


@typecheck(x=expr_int32, n=expr_int32, p=expr_float64,
           alternative=enumeration("two.sided", "greater", "less"))
def binom_test(x: Int32Expression, n: Int32Expression, p: Float64Expression, alternative: str) -> Float64Expression:
    """Performs a binomial test on `p` given `x` successes in `n` trials.

    Examples
    --------
    .. doctest::

        >>> hl.eval_expr(hl.binom_test(5, 10, 0.5, 'less'))
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
    return _func("binomTest", tfloat64, x, n, p, to_expr(alternative))


@typecheck(x=expr_float64, df=expr_float64)
def pchisqtail(x: Float64Expression, df: Float64Expression) -> Float64Expression:
    """Returns the probability under the right-tail starting at x for a chi-squared
    distribution with df degrees of freedom.

    Examples
    --------
    .. doctest::

        >>> hl.eval_expr(hl.pchisqtail(5, 1))
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
def pnorm(x: Float64Expression) -> Float64Expression:
    """The cumulative probability function of a standard normal distribution.

    Examples
    --------
    .. doctest::

        >>> hl.eval_expr(hl.pnorm(0))
        0.5

        >>> hl.eval_expr(hl.pnorm(1))
        0.8413447460685429

        >>> hl.eval_expr(hl.pnorm(2))
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
def ppois(x: Float64Expression,
          lamb: Float64Expression,
          lower_tail: BooleanExpression = True,
          log_p: BooleanExpression = False) -> Float64Expression:
    """The cumulative probability function of a Poisson distribution.

    Examples
    --------
    .. doctest::

        >>> hl.eval_expr(hl.ppois(2, 1))
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
def qchisqtail(p: Float64Expression, df: Float64Expression) -> Float64Expression:
    """Inverts :meth:`.pchisqtail`.

    Examples
    --------
    .. doctest::

        >>> hl.eval_expr(hl.qchisqtail(0.01, 1))
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
def qnorm(p: Float64Expression) -> Float64Expression:
    """Inverts :meth:`.pnorm`.

    Examples
    --------
    .. doctest::

        >>> hl.eval_expr(hl.qnorm(0.90))
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
def qpois(p: Float64Expression,
          lamb: Float64Expression,
          lower_tail: BooleanExpression = True,
          log_p: BooleanExpression = False) -> Float64Expression:
    """Inverts :meth:`.ppois`.

    Examples
    --------
    .. doctest::

        >>> hl.eval_expr(hl.qpois(0.99, 1))
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
def range(start: Int32Expression, stop: Int32Expression, step: Int32Expression = 1) -> ArrayNumericExpression:
    """Returns an array of integers from `start` to `stop` by `step`.

    Examples
    --------
    .. doctest::

        >>> hl.eval_expr(hl.range(0, 10))
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

        >>> hl.eval_expr(hl.range(0, 10, step=3))
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
def rand_bool(p: Float64Expression) -> BooleanExpression:
    """Returns ``True`` with probability `p` (RNG).

    Examples
    --------
    .. doctest::

        >>> hl.eval_expr(hl.rand_bool(0.5))
        True

        >>> hl.eval_expr(hl.rand_bool(0.5))
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
def rand_norm(mean: Float64Expression = 0, sd: Float64Expression = 1) -> Float64Expression:
    """Samples from a normal distribution with mean `mean` and standard deviation `sd` (RNG).

    Examples
    --------
    .. doctest::

        >>> hl.eval_expr(hl.rand_norm())
        1.5388475315213386

        >>> hl.eval_expr(hl.rand_norm())
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
def rand_pois(lamb: Float64Expression) -> Float64Expression:
    """Samples from a Poisson distribution with rate parameter `lamb` (RNG).

    Examples
    --------
    .. doctest::

        >>> hl.eval_expr(hl.rand_pois(1))
        2.0

        >>> hl.eval_expr(hl.rand_pois(1))
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
def rand_unif(min: Float64Expression, max: Float64Expression) -> Float64Expression:
    """Returns a random floating-point number uniformly drawn from the interval [`min`, `max`].

    Examples
    --------
    .. doctest::

        >>> hl.eval_expr(hl.rand_unif(0, 1))
        0.7983073825816226

        >>> hl.eval_expr(hl.rand_unif(0, 1))
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
def sqrt(x: Float64Expression) -> Float64Expression:
    """Returns the square root of `x`.

    Examples
    --------
    .. doctest::

        >>> hl.eval_expr(hl.sqrt(3))
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


@typecheck(ref=expr_str, alt=expr_str)
def is_snp(ref: StringExpression, alt: StringExpression) -> BooleanExpression:
    """Returns ``True`` if the alleles constitute a single nucleotide polymorphism.

    Examples
    --------
    .. doctest::

        >>> hl.eval_expr(hl.is_snp('A', 'T'))
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
    return _func("is_snp", tbool, ref, alt)


@typecheck(ref=expr_str, alt=expr_str)
def is_mnp(ref: StringExpression, alt: StringExpression) -> BooleanExpression:
    """Returns ``True`` if the alleles constitute a multiple nucleotide polymorphism.

    Examples
    --------
    .. doctest::

        >>> hl.eval_expr(hl.is_mnp('AA', 'GT'))
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
    return _func("is_mnp", tbool, ref, alt)


@typecheck(ref=expr_str, alt=expr_str)
def is_transition(ref: StringExpression, alt: StringExpression) -> BooleanExpression:
    """Returns ``True`` if the alleles constitute a transition.

    Examples
    --------
    .. doctest::

        >>> hl.eval_expr(hl.is_transition('A', 'T'))
        False

        >>> hl.eval_expr(hl.is_transition('A', 'G'))
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
    return _func("is_transition", tbool, ref, alt)


@typecheck(ref=expr_str, alt=expr_str)
def is_transversion(ref: StringExpression, alt: StringExpression) -> BooleanExpression:
    """Returns ``True`` if the alleles constitute a transversion.

    Examples
    --------
    .. doctest::

        >>> hl.eval_expr(hl.is_transition('A', 'T'))
        True

        >>> hl.eval_expr(hl.is_transition('A', 'G'))
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
    return _func("is_transversion", tbool, ref, alt)


@typecheck(ref=expr_str, alt=expr_str)
def is_insertion(ref: StringExpression, alt: StringExpression) -> BooleanExpression:
    """Returns ``True`` if the alleles constitute an insertion.

    Examples
    --------
    .. doctest::

        >>> hl.eval_expr(hl.is_insertion('A', 'ATT'))
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
    return _func("is_insertion", tbool, ref, alt)


@typecheck(ref=expr_str, alt=expr_str)
def is_deletion(ref: StringExpression, alt: StringExpression) -> BooleanExpression:
    """Returns ``True`` if the alleles constitute a deletion.

    Examples
    --------
    .. doctest::

        >>> hl.eval_expr(hl.is_deletion('ATT', 'A'))
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
    return _func("is_deletion", tbool, ref, alt)


@typecheck(ref=expr_str, alt=expr_str)
def is_indel(ref: StringExpression, alt: StringExpression) -> BooleanExpression:
    """Returns ``True`` if the alleles constitute an insertion or deletion.

    Examples
    --------
    .. doctest::

        >>> hl.eval_expr(hl.is_indel('ATT', 'A'))
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
    return _func("is_indel", tbool, ref, alt)


@typecheck(ref=expr_str, alt=expr_str)
def is_star(ref: StringExpression, alt: StringExpression) -> BooleanExpression:
    """Returns ``True`` if the alleles constitute an upstream deletion.

    Examples
    --------
    .. doctest::

        >>> hl.eval_expr(hl.is_deletion('A', '*'))
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
    return _func("is_star", tbool, ref, alt)


@typecheck(ref=expr_str, alt=expr_str)
def is_complex(ref: StringExpression, alt: StringExpression) -> BooleanExpression:
    """Returns ``True`` if the alleles constitute a complex polymorphism.

    Examples
    --------
    .. doctest::

        >>> hl.eval_expr(hl.is_deletion('ATT', 'GCA'))
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
    return _func("is_complex", tbool, ref, alt)


@typecheck(ref=expr_str, alt=expr_str)
def is_strand_ambiguous(ref: StringExpression, alt: StringExpression) -> BooleanExpression:
    """Returns ``True`` if the alleles are strand ambiguous.

    Strand ambiguous allele pairs are ``A/T``, ``T/A``,
    ``C/G``, and ``G/C`` where the first allele is `ref`
    and the second allele is `alt`.

    Examples
    --------
    .. doctest::

        >>> hl.eval_expr(hl.is_strand_ambiguous('A', 'T'))
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
def allele_type(ref: StringExpression, alt: StringExpression) -> BooleanExpression:
    """Returns the type of the polymorphism as a string.

    Examples
    --------
    .. doctest::

        >>> hl.eval_expr(hl.allele_type('A', 'T'))
        'SNP'

        >>> hl.eval_expr(hl.allele_type('ATT', 'A'))
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
    return _func("allele_type", tstr, ref, alt)


@typecheck(s1=expr_str, s2=expr_str)
def hamming(s1: StringExpression, s2: StringExpression) -> Int32Expression:
    """Returns the Hamming distance between the two strings.

    Examples
    --------
    .. doctest::

        >>> hl.eval_expr(hl.hamming('ATATA', 'ATGCA'))
        2

        >>> hl.eval_expr(hl.hamming('abcdefg', 'zzcdefz'))
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


@typecheck(x=expr_any)
def str(x: Expression) -> StringExpression:
    """Returns the string representation of `x`.

    Examples
    --------
    .. doctest::

        >>> hl.eval_expr(hl.str(hl.struct(a=5, b=7)))
        '{"a":5,"b":7}'

    Parameters
    ----------
    x

    Returns
    -------
    :class:`.StringExpression`
    """
    return _func("str", tstr, x)


@typecheck(c=expr_call, i=expr_int32)
def downcode(c: CallExpression, i: Int32Expression) -> CallExpression:
    """Create a new call by setting all alleles other than i to ref

    Examples
    --------
    Preserve the third allele and downcode all other alleles to reference.

    .. doctest::

        >>> hl.eval_expr(hl.downcode(hl.call(1, 2), 2))
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
def gq_from_pl(pl: ArrayNumericExpression) -> Int32Expression:
    """Compute genotype quality from Phred-scaled probability likelihoods.

    Examples
    --------
    .. doctest::

        >>> hl.eval_expr(hl.gq_from_pl([0,69,1035]))
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
def triangle(n: Int32Expression) -> Int32Expression:
    """Returns the triangle number of `n`.

    Examples
    --------
    .. doctest::

        >>> hl.eval_expr(hl.triangle(3))
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
def filter(f: Callable[[Expression], BooleanExpression], collection: Coll_T) -> Coll_T:
    """Returns a new collection containing elements where `f` returns ``True``.

    Examples
    --------
    .. doctest::

        >>> a = [1, 2, 3, 4]
        >>> s = {'Alice', 'Bob', 'Charlie'}

        >>> hl.eval_expr(hl.filter(lambda x: x % 2 == 0, a))
        [2, 4]

        >>> hl.eval_expr(hl.filter(lambda x: ~(x[-1] == 'e'), s))
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
def any(f: Callable[[Expression], BooleanExpression],
        collection: CollectionExpression) -> BooleanExpression:
    """Returns ``True`` if `f` returns ``True`` for any element.

    Examples
    --------
    .. doctest::

        >>> a = ['The', 'quick', 'brown', 'fox']
        >>> s = {1, 3, 5, 6, 7, 9}

        >>> hl.eval_expr(hl.any(lambda x: x[-1] == 'x', a))
        True

        >>> hl.eval_expr(hl.any(lambda x: x % 4 == 0, s))
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
def all(f: Callable[[Expression], BooleanExpression],
        collection: CollectionExpression) -> BooleanExpression:
    """Returns ``True`` if `f` returns ``True`` for every element.

    Examples
    --------
    .. doctest::

        >>> a = ['The', 'quick', 'brown', 'fox']
        >>> s = {1, 3, 5, 6, 7, 9}

        >>> hl.eval_expr(hl.all(lambda x: hl.len(x) > 3, a))
        False

        >>> hl.eval_expr(hl.all(lambda x: x < 10, s))
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
def find(f: Callable[[Expression], BooleanExpression], collection: CollectionExpression) -> Expression:
    """Returns the first element where `f` returns ``True``.

    Examples
    --------
    .. doctest::

        >>> a = ['The', 'quick', 'brown', 'fox']
        >>> s = {1, 3, 5, 6, 7, 9}

        >>> hl.eval_expr(hl.find(lambda x: x[-1] == 'x', a))
        'fox'

        >>> hl.eval_expr(hl.find(lambda x: x % 4 == 0, s))
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

    return collection._bin_lambda_method("find", f,
                                         collection.dtype.element_type,
                                         lambda _: collection.dtype.element_type)


@typecheck(f=func_spec(1, expr_any),
           collection=expr_oneof(expr_set(), expr_array()))
def flatmap(f: Callable[[Expression], Coll_T], collection: Coll_T) -> Coll_T:
    """Map each element of the collection to a new collection, and flatten the results.

    Examples
    --------
    .. doctest::

        >>> a = [[0, 1], [1, 2], [4, 5, 6, 7]]

        >>> hl.eval_expr(hl.flatmap(lambda x: x[1:], a))
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
def group_by(f: Callable[[Expression], Expression], collection: CollectionExpression) -> DictExpression:
    """Group collection elements into a dict according to a lambda function.

    Examples
    --------
    .. doctest::

        >>> a = ['The', 'quick', 'brown', 'fox']

        >>> hl.eval_expr(hl.group_by(lambda x: hl.len(x), a))
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
def zip(*arrays: ArrayExpression, fill_missing: bool = False) -> ArrayExpression:
    """Zip together arrays into a single array.

    Examples
    --------
    .. doctest::

        >>> hl.eval_expr(hl.zip([1], [10, 20], [100, 200, 300]))
        [(1, 10, 100)]

        >>> hl.eval_expr(hl.zip([1], [10, 20], [100, 200, 300], fill_missing=True))
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

        return bind([hl.len(a) for a in arrays], _)
    else:
        def _(array_lens):
            result_len = hl.min(array_lens)
            indices = hl.range(0, result_len)
            return hl.map(lambda i: builtins.tuple(arrays[j][i] for j in builtins.range(n_arrays)), indices)

        return bind([hl.len(a) for a in arrays], _)


@typecheck(f=func_spec(1, expr_any),
           collection=expr_oneof(expr_set(), expr_array()))
def map(f: Callable[[Expression], Expression], collection: Coll_T) -> Coll_T:
    """Transform each element of a collection.

    Examples
    --------
    .. doctest::

        >>> a = ['The', 'quick', 'brown', 'fox']

        >>> hl.eval_expr(hl.map(lambda x: hl.len(x), a))
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
def len(x: Union[CollectionExpression, DictExpression, StringExpression, TupleExpression,
                 StructExpression]) -> Int32Expression:
    """Returns the size of a collection or string.

    Examples
    --------
    .. doctest::

        >>> a = ['The', 'quick', 'brown', 'fox']
        >>> s = {1, 3, 5, 6, 7, 9}

        >>> hl.eval_expr(hl.len(a))
        4

        >>> hl.eval_expr(hl.len(s))
        6

        >>> hl.eval_expr(hl.len("12345"))
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
def max(*exprs: Union[CollectionExpression, Num_T]) -> Num_T:
    """Returns the maximum element of a collection or of given numeric expressions.

    Examples
    --------
    .. doctest::

        Take the maximum value of an array:

        >>> hl.eval_expr(hl.max([1, 3, 5, 6, 7, 9]))
        9

        Take the maximum value of values:

        >>> hl.eval_expr(hl.max(1, 50, 2))
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
def min(*exprs: Union[CollectionExpression, Num_T]) -> Num_T:
    """Returns the minimum of a collection or of given numeric expressions.

    Examples
    --------
    .. doctest::

        Take the minimum value of an array:

        >>> hl.eval_expr(hl.max([2, 3, 5, 6, 7, 9]))
        2

        Take the minimum value:

        >>> hl.eval_expr(hl.max(12, 50, 2))
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
def abs(x: Num_T) -> Num_T:
    """Take the absolute value of a numeric value or array.

    Examples
    --------
    .. doctest::

        >>> hl.eval_expr(hl.abs(-5))
        5

        >>> hl.eval_expr(hl.abs([1.0, -2.5, -5.1]))
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
def signum(x: NumericExpression) -> Int32Expression:
    """Returns the sign (1, 0, or -1) of a numeric value or array.

    Examples
    --------
    .. doctest::

        >>> hl.eval_expr(hl.signum(-1.23))
        -1

        >>> hl.eval_expr(hl.signum(555))
        1

        >>> hl.eval_expr(hl.signum(0.0))
        0

        >>> hl.eval_expr(hl.signum([1, -5, 0, -125]))
        [1, -1, 0, -1]

    Parameters
    ----------
    x : :class:`.NumericExpression` or :class:`.ArrayNumericExpression`

    Returns
    -------
    :class:`.NumericExpression` or :class:`.ArrayNumericExpression`.
    """
    if isinstance(x.dtype, tarray):
        return map(signum, x)
    else:
        return x._method('signum', tint32)


@typecheck(collection=expr_oneof(expr_set(expr_numeric), expr_array(expr_numeric)))
def mean(collection: CollectionExpression) -> Float64Expression:
    """Returns the mean of all values in the collection.

    Examples
    --------
    .. doctest::

        >>> a = [1, 3, 5, 6, 7, 9]

        >>> hl.eval_expr(hl.mean(a))
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
def median(collection: CollectionExpression) -> Float64Expression:
    """Returns the median value in the collection.

    Examples
    --------
    .. doctest::

        >>> a = [1, 3, 5, 6, 7, 9]

        >>> hl.eval_expr(hl.median(a))
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
def product(collection: CollectionExpression) -> Float64Expression:
    """Returns the product of values in the collection.

    Examples
    --------
    .. doctest::

        >>> a = [1, 3, 5, 6, 7, 9]

        >>> hl.eval_expr(hl.product(a))
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
def sum(collection: CollectionExpression) -> Float64Expression:
    """Returns the sum of values in the collection.

    Examples
    --------
    .. doctest::

        >>> a = [1, 3, 5, 6, 7, 9]

        >>> hl.eval_expr(hl.sum(a))
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
def struct(**kwargs: Expression) -> StructExpression:
    """Construct a struct expression.

    Examples
    --------
    .. doctest::

        >>> s = hl.struct(a=5, b='Foo')

        >>> hl.eval_expr(s.a)
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
    .. doctest::

        >>> t = hl.tuple([1, 2, '3'])
        >>> hl.eval_expr(t)
        (1, 2, '3')

        >>> hl.eval_expr(t[2])
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
def set(collection: Union[CollectionExpression, Set, List]) -> SetExpression:
    """Convert a set expression.

    Examples
    --------
    .. doctest::

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
    .. doctest::

        >>> hl.eval_expr(hl.empty_set(hl.tstr))
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
def array(collection: Union[CollectionExpression, DictExpression, Set, List, Dict]) -> ArrayExpression:
    """Construct an array expression.

    Examples
    --------
    .. doctest::

        >>> s = {'Bob', 'Charlie', 'Alice'}

        >>> hl.eval_expr(hl.array(s))
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
    .. doctest::

        >>> hl.eval_expr(hl.empty_array(hl.tint32))
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
    .. doctest::

        >>> hl.eval_expr(hl.empty_dict(hl.tstr, hl.tint32))
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
def flatten(collection: Coll_T) -> Coll_T:
    """Flatten a nested collection by concatenating sub-collections.

    Examples
    --------
    .. doctest::

        >>> a = [[1, 2], [2, 3]]

        >>> hl.eval_expr(hl.flatten(a))
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
def delimit(collection: CollectionExpression, delimiter: StringExpression = ',') -> StringExpression:
    """Joins elements of `collection` into single string delimited by `delimiter`.

    Examples
    --------
    .. doctest::

        >>> a = ['Bob', 'Charlie', 'Alice', 'Bob', 'Bob']

        >>> hl.eval_expr(hl.delimit(a))
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
def sorted(collection: ArrayExpression,
           key: Optional[Callable[[Expression], Expression]] = None,
           reverse: BooleanExpression = False) -> ArrayExpression:
    """Returns a sorted array.

    Examples
    --------
    .. doctest::

        >>> a = ['Charlie', 'Alice', 'Bob']

        >>> hl.eval_expr(hl.sorted(a))
        ['Alice', 'Bob', 'Charlie']

        >>> hl.eval_expr(hl.sorted(a, reverse=False))
        ['Charlie', 'Bob', 'Alice']

        >>> hl.eval_expr(hl.sorted(a, key=lambda x: hl.len(x)))
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
    :class:`.ArrayNumericExpression`
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
def argmin(array: ArrayNumericExpression, unique: bool = False) -> Int32Expression:
    """Return the index of the minimum value in the array.

    Examples
    --------

    .. doctest::

        >>> hl.eval_expr(hl.argmin([0.2, 0.3, 0.6]))
        0

        >>> hl.eval_expr(hl.argmin([0.4, 0.2, 0.2]))
        1

        >>> hl.eval_expr(hl.argmin([0.4, 0.2, 0.2], unique=True))
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
def argmax(array: ArrayNumericExpression, unique: bool = False) -> Int32Expression:
    """Return the index of the maximum value in the array.

    Examples
    --------

    .. doctest::

        >>> hl.eval_expr(hl.argmax([0.2, 0.2, 0.6]))
        2

        >>> hl.eval_expr(hl.argmax([0.4, 0.4, 0.2]))
        0

        >>> hl.eval_expr(hl.argmax([0.4, 0.4, 0.2], unique=True))
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


@typecheck(expr=expr_oneof(expr_numeric, expr_bool, expr_str))
def float64(expr: Union[int, float, str, NumericExpression, BooleanExpression, StringExpression]) -> Float64Expression:
    """Convert to a 64-bit floating point expression.

    Examples
    --------
    .. doctest::

        >>> hl.eval_expr(hl.float64('1.1'))
        1.1

        >>> hl.eval_expr(hl.float64(1))
        1.0

        >>> hl.eval_expr(hl.float64(True))
        1.0

    Parameters
    ----------
    expr : :class:`.NumericExpression` or :class:`.BooleanExpression` or :class:`.StringExpression`

    Returns
    -------
    :class:`.NumericExpression` of type :py:data:`.tfloat64`
    """
    return expr._method("toFloat64", tfloat64)


@typecheck(expr=expr_oneof(expr_numeric, expr_bool, expr_str))
def float32(expr: Union[int, float, str, NumericExpression, BooleanExpression, StringExpression]) -> Float32Expression:
    """Convert to a 32-bit floating point expression.

    Examples
    --------
    .. doctest::

        >>> hl.eval_expr(hl.float32('1.1'))
        1.1

        >>> hl.eval_expr(hl.float32(1))
        1.0

        >>> hl.eval_expr(hl.float32(True))
        1.0

    Parameters
    ----------
    expr : :class:`.NumericExpression` or :class:`.BooleanExpression` or :class:`.StringExpression`

    Returns
    -------
    :class:`.NumericExpression` of type :py:data:`.tfloat32`
    """
    return expr._method("toFloat32", tfloat32)


@typecheck(expr=expr_oneof(expr_numeric, expr_bool, expr_str))
def int64(expr: Union[int, float, str, NumericExpression, BooleanExpression, StringExpression]) -> Int64Expression:
    """Convert to a 64-bit integer expression.

    Examples
    --------
    .. doctest::

        >>> hl.eval_expr(hl.int64('1'))
        1

        >>> hl.eval_expr(hl.int64(1.5))
        1

        >>> hl.eval_expr(hl.int64(True))
        1

    Parameters
    ----------
    expr : :class:`.NumericExpression` or :class:`.BooleanExpression` or :class:`.StringExpression`

    Returns
    -------
    :class:`.NumericExpression` of type :py:data:`.tint64`
    """
    return expr._method("toInt64", tint64)


@typecheck(expr=expr_oneof(expr_numeric, expr_bool, expr_str))
def int32(expr: Union[int, float, str, NumericExpression, BooleanExpression, StringExpression]) -> Int32Expression:
    """Convert to a 32-bit integer expression.

    Examples
    --------
    .. doctest::

        >>> hl.eval_expr(hl.int32('1'))
        1

        >>> hl.eval_expr(hl.int32(1.5))
        1

        >>> hl.eval_expr(hl.int32(True))
        1

    Parameters
    ----------
    expr : :class:`.NumericExpression` or :class:`.BooleanExpression` or :class:`.StringExpression`

    Returns
    -------
    :class:`.NumericExpression` of type :py:data:`.tint32`
    """
    return expr._method("toInt32", tint32)


@typecheck(expr=expr_oneof(expr_numeric, expr_bool, expr_str))
def int(expr: Union[int, float, str, NumericExpression, BooleanExpression, StringExpression]) -> Int32Expression:
    """Convert to a 32-bit integer expression.

    Examples
    --------
    .. doctest::

        >>> hl.eval_expr(hl.int('1'))
        1

        >>> hl.eval_expr(hl.int(1.5))
        1

        >>> hl.eval_expr(hl.int(True))
        1

    Note
    ----
    Alias for :func:`.int32`.

    Parameters
    ----------
    expr : :class:`.NumericExpression` or :class:`.BooleanExpression` or :class:`.StringExpression`

    Returns
    -------
    :class:`.NumericExpression` of type :py:data:`.tint32`
    """
    return int32(expr)


@typecheck(expr=expr_oneof(expr_numeric, expr_bool, expr_str))
def float(expr: Union[int, float, str, NumericExpression, BooleanExpression, StringExpression]) -> Float64Expression:
    """Convert to a 64-bit floating point expression.

    Examples
    --------
    .. doctest::

        >>> hl.eval_expr(hl.float('1.1'))
        1.1

        >>> hl.eval_expr(hl.float(1))
        1.0

        >>> hl.eval_expr(hl.float(True))
        1.0

    Note
    ----
    Alias for :func:`.float64`.

    Parameters
    ----------
    expr : :class:`.NumericExpression` or :class:`.BooleanExpression` or :class:`.StringExpression`

    Returns
    -------
    :class:`.NumericExpression` of type :py:data:`.tfloat64`
    """
    return float64(expr)


@typecheck(expr=expr_oneof(expr_numeric, expr_bool, expr_str))
def bool(expr: Union[int, float, str, NumericExpression, BooleanExpression, StringExpression]) -> BooleanExpression:
    """Convert to a Boolean expression.

    Examples
    --------
    .. doctest::

        >>> hl.eval_expr(hl.bool('TRUE'))
        True

        >>> hl.eval_expr(hl.bool(1.5))
        True

    Notes
    -----
    Numeric expressions return ``True`` if they are non-zero, and ``False``
    if they are zero.

    Acceptable string values are: ``'True'``, ``'true'``, ``'TRUE'``,
    ``'False'``, ``'false'``, and ``'FALSE'``.

    Returns
    -------
    :class:`.BooleanExpression`
    """
    if is_numeric(expr.dtype):
        return expr != 0
    else:
        return expr._method("toBoolean", tbool)


@typecheck(reference_genome=reference_genome_type,
           contig=expr_str,
           position=expr_int32,
           before=expr_int32,
           after=expr_int32)
def get_sequence(reference_genome, contig, position, before=0, after=0):
    """Return the reference sequence at a given locus.

    Examples
    --------

    Return the reference allele for ``'GRCh37'`` at the locus ``'1:45323'``:

    .. doctest::
        :options: +SKIP

        >>> hl.get_sequence('GRCh37', '1', 45323)
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
    reference_genome : :obj:`str` or :class:`.ReferenceGenome`
        Reference genome to use. Must have a reference sequence available.
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

    Returns
    -------
    :class:`.Expression` of type :py:data:`.tstr`
    """

    if not reference_genome.has_sequence():
        raise TypeError("Reference genome '{}' does not have a sequence loaded. Use 'add_sequence' to load the sequence from a FASTA file.".format(reference_genome.name))
    return _func("getReferenceSequence({})".format(reference_genome.name), tstr,
                 contig, position, before, after)
