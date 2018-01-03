from __future__ import print_function  # Python 2 and 3 print compatibility
from hail.typecheck import *
from hail.expr.expression import *
from hail.expr.ast import *
from hail.genetics import Variant, Locus, Call, GenomeReference

def _func(name, ret_type, *args):
    indices, aggregations, joins = unify_all(*args)
    return construct_expr(ApplyMethod(name, *(a._ast for a in args)), ret_type, indices, aggregations, joins)

@typecheck(t=Type)
def null(t):
    """Creates an expression representing a missing value of a specified type.

    Examples
    --------
    .. doctest::

        >>> eval_expr(functions.null(TString()))
        None

    Notes
    -----
    This method is useful for constructing an expression that includes missing
    values, since :obj:`None` cannot be interpreted as an expression.

    Parameters
    ----------
    t : :py:class:`hail.expr.Type`
        Type of the missing expression.

    Returns
    -------
    :py:class:`hail.expr.expression.Expression`
        A missing expression of type `t`.
    """
    return construct_expr(Literal('NA: {}'.format(t)), t)


def capture(x):
    """Captures a Python variable or object as an expression.

    Examples
    --------
    .. doctest::

        >>> eval_expr(functions.capture(5))
        5

        >>> eval_expr(functions.capture([1, 2, 3]))
        [1, 2, 3]

    Warning
    -------
    For large objects, use :py:meth:`hail.expr.functions.broadcast`.

    Parameters
    ----------
    x
        Python variable to capture as an expression.

    Returns
    -------
    :py:class:`hail.expr.expression.Expression`
        An expression representing `x`.
    """
    return to_expr(x)


def broadcast(x):
    """Broadcasts a Python variable or object as an expression.

    Examples
    --------
    .. doctest::

        >>> table = Table.range(8)
        >>> greetings = functions.broadcast({1: 'Good morning', 4: 'Good afternoon', 6 : 'Good evening'})
        >>> table.annotate(greeting = greetings.get(table.index)).show()
        +-------+----------------+
        | index | greeting       |
        +-------+----------------+
        | Int32 | String         |
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
    :class:hail.api2.Table or :class:hail.api2.MatrixTable.

    Parameters
    ----------
    x
        Python variable to broadcast as an expression.

    Returns
    -------
    :py:class:`hail.expr.expression.Expression`
        An expression representing `x`.
    """
    expr = to_expr(x)
    uid = Env._get_uid()

    def joiner(obj):
        from hail.api2.table import Table
        from hail.api2.matrixtable import MatrixTable
        if isinstance(obj, Table):
            return Table(obj._jt.annotateGlobalExpr('{} = {}'.format(uid, expr._ast.to_hql())))
        else:
            assert isinstance(obj, MatrixTable)
            return MatrixTable(obj._jvds.annotateGlobalExpr('global.{} = {}'.format(uid, expr._ast.to_hql())))

    return construct_expr(GlobalJoinReference(uid), expr._type, joins=(Join(joiner, [uid]),))


@typecheck(predicate=expr_bool, then_case=anytype, else_case=anytype)
@args_to_expr
def cond(predicate, then_case, else_case):
    """Expression for an if/else statement; tests a predicate and returns one of two options based on the result.

    Examples
    --------
    .. doctest::

        >>> x = 5
        >>> eval_expr( functions.cond(x < 2, 'Hi', 'Bye') )
        'Bye'

        >>> a = functions.capture([1, 2, 3, 4])
        >>> eval_expr( functions.cond(a.length() > 0,
        ...                   2.0 * a,
        ...                   a / 2.0) )
        [2.0, 4.0, 6.0, 8.0]

    Notes
    -----

    If `predicate` evaluates to ``True``, returns `then_case`. If `predicate`
    evaluates to ``False``, returns `else_case`. If `predicate` is missing, returns
    missing.

    Note
    ----
    The type of `then_case` and `else_case` must be the same.

    Parameters
    ----------
    predicate : bool or :py:class:`.hail.expr.expression.BooleanExpression`
        Predicate to test.
    then_case
        Branch to return if the predicate is true.
    else_case
        Branch to return if the predicate is false.

    Returns
    -------
    :py:class:`hail.expr.expression.Expression`
        `then_case`, `else_case`, or missing.
    """
    indices, aggregations, joins = unify_all(predicate, then_case, else_case)
    # TODO: promote types
    return construct_expr(Condition(predicate._ast, then_case._ast, else_case._ast),
                          then_case._type, indices, aggregations, joins)

def bind(expr, f):
    """Bind a temporary variable and use it in a function.

    Examples
    --------
    Expressions are "inlined", leading to perhaps unexpected behavior
    when randomness is involved. For example, let us define a variable
    `x` from the :meth:`rand_unif` method:

    >>> x = functions.rand_unif(0, 1)

    Note that evaluating `x` multiple times returns different results.
    The value of evaluating `x` is unknown when the expression is defined.

    .. doctest::

        >>> eval_expr(x)
        0.3189309481038456

        >>> eval_expr(x)
        0.20842918568366375

    What if we evaluate `x` multiple times in the same invocation of
    :meth:`hail.expr.eval_expr`?

    .. doctest::

        >>> eval_expr([x, x, x])
        [0.49582541026815163, 0.8549329234134524, 0.7016124997911775]

    The random number generator is called separately for each inclusion
    of `x`. This method, `bind`, is the solution to this problem!

    .. doctest::

        >>> eval_expr(functions.bind(x, lambda y: [y, y, y]))
        [0.7897028763765286, 0.7897028763765286, 0.7897028763765286]

    Parameters
    ----------
    expr : :class:`Expression`
        Expression to bind.
    f : callable
        Function of `expr`.

    Returns
    -------
    :class:`Expression`
        Result of evaluating `f` with `expr` as an argument.
    """
    uid = Env._get_uid()
    expr = to_expr(expr)

    f_input = construct_expr(Reference(uid), expr._type, expr._indices, expr._aggregations, expr._joins)
    lambda_result = to_expr(f(f_input))

    indices, aggregations, joins = unify_all(expr, lambda_result)
    ast = Bind(uid, expr._ast, lambda_result._ast)
    return construct_expr(ast, lambda_result._type, indices, aggregations, joins)


@args_to_expr
@typecheck(c1=expr_int32, c2=expr_int32, c3=expr_int32, c4=expr_int32)
def chisq(c1, c2, c3, c4):
    """Calculates p-value (Chi-square approximation) and odds ratio for a 2x2 table.

    Examples
    --------
    .. doctest::

        >>> eval_expr(functions.chisq(10, 10,
        ...                   10, 10))
        Struct(oddsRatio=1.0, pValue=1.0)

        >>> eval_expr(functions.chisq(30, 30,
        ...                   50, 10))
        Struct(oddsRatio=0.2, pValue=0.000107511176729)

    Parameters
    ----------
    c1 : int or :py:class:`hail.expr.expression.Int32Expression`
        Value for cell 1.
    c2 : int or :py:class:`hail.expr.expression.Int32Expression`
        Value for cell 2.
    c3 : int or :py:class:`hail.expr.expression.Int32Expression`
        Value for cell 3.
    c4 : int or :py:class:`hail.expr.expression.Int32Expression`
        Value for cell 4.

    Returns
    -------
    :py:class:`hail.expr.expression.StructExpression`
        A struct expression with two fields, `pValue` (``Float64``) and `oddsRatio` (``Float64``).
    """
    ret_type = TStruct(['pValue', 'oddsRatio'], [TFloat64(), TFloat64()])
    return _func("chisq", ret_type, c1, c2, c3, c4)


@args_to_expr
@typecheck(left=expr_variant, right=expr_variant)
def combine_variants(left, right):
    """Combines the alleles of two variants at the same locus to form a new variant.

    This method ensures that the resulting alleles are represented uniformly and minimally.
    In addition to the resulting variant containing all alleles, this function also returns
    the mapping from the old to the new allele indices. Note that this mapping counts the
    reference allele, always contains the reference allele mapping 0 -> 0.

    Parameters
    ----------
    left : :py:class:`hail.genetics.Variant` or :py:class:`hail.expr.expression.VariantExpression`
        First variant.
    right : :py:class:`hail.genetics.Variant` or :py:class:`hail.expr.expression.VariantExpression`
        Second variant.

    Returns
    -------
    :py:class:`hail.expr.expression.StructExpression`
        A struct expression with three fields, `variant` (``Variant``), `laIndices` (``Dict[Int32, Int32]``),
        and `raIndices` (``Dict[Int32, Int32]``)
    """
    if not left._type._rg == right._type._rg:
        raise TypeError('Reference genome mismatch: {}, {}'.format(left._type._rg, right._type._rg))
    ret_type = TStruct(['variant', 'laIndices', 'raIndices'],
                       [left._type, TDict(TInt32(), TInt32()), TDict(TInt32(), TInt32())])
    return _func("combineVariants", ret_type, left, right)


@typecheck(c1=expr_int32, c2=expr_int32, c3=expr_int32, c4=expr_int32, min_cell_count=expr_int32)
@args_to_expr
def ctt(c1, c2, c3, c4, min_cell_count):
    """Calculates p-value and odds ratio for 2x2 table.

    Examples
    --------
    .. doctest::

        >>> eval_expr(functions.ctt(10, 10,
        ...                 10, 10, min_cell_count=15))
        Struct(oddsRatio=1.0, pValue=1.0)

        >>> eval_expr(functions.ctt(30, 30,
        ...                 50, 10, min_cell_count=15))
        Struct(oddsRatio=0.202874620964, pValue=0.000190499944324)

    Notes
    -----
     If any cell is lower than `min_cell_count`, Fisher's exact test is used. Otherwise, faster
     chi-squared approximation is used.

    Parameters
    ----------
    c1 : int or :py:class:`hail.expr.expression.Int32Expression`
        Value for cell 1.
    c2 : int or :py:class:`hail.expr.expression.Int32Expression`
        Value for cell 2.
    c3 : int or :py:class:`hail.expr.expression.Int32Expression`
        Value for cell 3.
    c4 : int or :py:class:`hail.expr.expression.Int32Expression`
        Value for cell 4.
    min_cell_count : int or :py:class:`hail.expr.expression.Int32Expression`
        Minimum cell count for chi-squared approximation.

    Returns
    -------
    :py:class:`hail.expr.expression.StructExpression`
        A struct expression with two fields, `pValue` (``Float64``) and `oddsRatio` (``Float64``).
    """
    ret_type = TStruct(['pValue', 'oddsRatio'], [TFloat64(), TFloat64()])
    return _func("ctt", ret_type, c1, c2, c3, c4, min_cell_count)


@typecheck(keys=expr_list, values=expr_list)
@args_to_expr
def Dict(keys, values):
    """Creates a dictionary from a list of keys and a list of values.

    Examples
    --------
    .. doctest::

        >>> eval_expr(functions.Dict(['foo', 'bar', 'baz'], [1, 2, 3]))
        {u'bar': 2, u'baz': 3, u'foo': 1}

    Notes
    -----
    `keys` and `values` must be have the same length.

    Parameters
    ----------
    keys : list or :py:class:`hail.expr.expression.ArrayExpression`
        The keys of the resulting dictionary.
    values : list or :py:class:`hail.expr.expression.Int32Expression`
        The values of the resulting dictionary.

    Returns
    -------
    :py:class:`hail.expr.expression.DictExpression`
        A dictionary expression constructed from `keys` and `values`.

    """
    key_col = to_expr(keys)
    value_col = to_expr(values)
    ret_type = TDict(key_col._type, value_col._type)
    return _func("Dict", ret_type, keys, values)


@typecheck(x=expr_numeric, lamb=expr_numeric, log_p=expr_bool)
@args_to_expr
def dpois(x, lamb, log_p=False):
    """Compute the (log) probability density at x of a Poisson distribution with rate parameter `lamb`.

    Examples
    --------
    .. doctest::

        >>> eval_expr(functions.dpois(5, 3))
        0.10081881344492458

    Parameters
    ----------
    x : float or :py:class:`hail.expr.expression.Float64Expression`
        Non-negative number at which to compute the probability density.
    lamb : float or :py:class:`hail.expr.expression.Float64Expression`
        Poisson rate parameter. Must be non-negative.
    log_p : bool or :py:class:`hail.expr.expression.BooleanExpression`
        If true, the natural logarithm of the probability density is returned.

    Returns
    -------
    :py:class:`hail.expr.expression.Float64Expression`
        The (log) probability density.
    """
    return _func("dpois", TFloat64(), x, lamb, log_p)


@typecheck(s=oneof(Struct, StructExpression), identifiers=tupleof(expr_str))
def drop(s, *identifiers):
    s = to_expr(s)
    ret_type = s._type._drop(*identifiers)
    return construct_expr(StructOp('drop', s._ast, *identifiers),
                          ret_type, s._indices, s._aggregations, s._joins)


@typecheck(x=expr_numeric)
@args_to_expr
def exp(x):
    """Computes `e` raised to the power `x`.

    Examples
    --------
    .. doctest::

        >>> eval_expr(functions.exp(2))
        7.38905609893065

    Parameters
    ----------
    x : float or :py:class:`hail.expr.expression.Float64Expression`

    Returns
    -------
    :py:class:`hail.expr.expression.Float64Expression`
    """
    return _func("exp", TFloat64(), x)


@typecheck(c1=expr_int32, c2=expr_int32, c3=expr_int32, c4=expr_int32)
@args_to_expr
def fisher_exact_test(c1, c2, c3, c4):
    """Calculates the p-value, odds ratio, and 95% confidence interval with Fisher's exact test for a 2x2 table.

    Examples
    --------
    .. doctest::

        >>> eval_expr(functions.fisher_exact_test(10, 10,
        ...                               10, 10))
        Struct(oddsRatio=1.0, pValue=1.0)

        >>> eval_expr(functions.fisher_exact_test(30, 30,
        ...                               50, 10))
        Struct(oddsRatio=0.202874620964, pValue=0.000190499944324)

    Notes
    -----
    This method is identical to the version implemented in
    `R <https://stat.ethz.ch/R-manual/R-devel/library/stats/html/fisher.test.html>`_ with default
    parameters (two-sided, alpha = 0.05, null hypothesis that the odds ratio equals 1).

    Parameters
    ----------
    c1 : int or :py:class:`hail.expr.expression.Int32Expression`
        Value for cell 1.
    c2 : int or :py:class:`hail.expr.expression.Int32Expression`
        Value for cell 2.
    c3 : int or :py:class:`hail.expr.expression.Int32Expression`
        Value for cell 3.
    c4 : int or :py:class:`hail.expr.expression.Int32Expression`
        Value for cell 4.

    Returns
    -------
    :py:class:`hail.expr.expression.StructExpression`
        A struct expression with four fields, `pValue` (``Float64``), `oddsRatio` (``Float64``),
        ci95Lower (``Float64``), and ci95Upper(``Float64``).
    """
    ret_type = TStruct(['pValue', 'oddsRatio', 'ci95Lower', 'ci95Upper'],
                       [TFloat64(), TFloat64(), TFloat64(), TFloat64()])
    return _func("fet", ret_type, c1, c2, c3, c4)


@typecheck(j=expr_int32, k=expr_int32)
@args_to_expr
def gt_index(j, k):
    """Convert from `j`/`k` pair to call index (the triangular number).

    Examples
    --------
    .. doctest::

        >>> eval_expr(functions.gt_index(0, 1))
        1

        >>> eval_expr(functions.gt_index(1, 2))
        4

    Notes
    -----
    The call index is given by ``(k * (k + 1) / 2) + j`` if ``j <= k`` and
    ``(j * (j + 1)) / 2 + k`` if ``k <= j``.

    Parameters
    ----------
    j : int or :py:class:`hail.expr.expression.Int32Expression`
        First diploid allele index.
    k : int or :py:class:`hail.expr.expression.Int32Expression`
        Second diploid allele index.

    Returns
    -------
    :py:class:`hail.expr.expression.Int32Expression`
    """
    return _func("gtIndex", TInt32(), j, k)


@typecheck(num_hom_ref=expr_int32, num_het=expr_int32, num_hom_var=expr_int32)
@args_to_expr
def hardy_weinberg_p(num_hom_ref, num_het, num_hom_var):
    """Compute Hardy-Weinberg Equilbrium p-value and heterozygosity ratio.

    Examples
    --------
    .. doctest::

        >>> eval_expr(functions.hardy_weinberg_p(20, 50, 26))
        Struct(rExpectedHetFrequency=0.500654450262, pHWE=0.762089599352)

        >>> eval_expr(functions.hardy_weinberg_p(37, 200, 85))
        Struct(rExpectedHetFrequency=0.489649643074, pHWE=1.13372103832e-06)

    Notes
    -----
    For more information, see the
    `Wikipedia page <https://en.wikipedia.org/wiki/Hardy%E2%80%93Weinberg_principle>`__

    Parameters
    ----------
    num_hom_ref : int or :py:class:`hail.expr.expression.Int32Expression`
        Homozygous reference count.
    num_het : int or :py:class:`hail.expr.expression.Int32Expression`
        Heterozygote count.
    num_hom_var : int or :py:class:`hail.expr.expression.Int32Expression`
        Homozygous alternate count.

    Returns
    -------
    :py:class:`hail.expr.expression.StructExpression`
        A struct expression with two fields, `rExpectedHetFrequency` (``Float64``) and`pValue` (``Float64``).
    """
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
    """Construct a locus expression from a chromosome and position.

    Examples
    --------
    .. doctest::

        >>> eval_expr(functions.locus("1", 10000))
        Locus(contig=1, position=10000, reference_genome=GRCh37)

    Parameters
    ----------
    contig : str or :py:class:`hail.expr.expression.StringExpression`
        Chromosome.
    pos : int or :py:class:`hail.expr.expression.Int32Expression`
        Base position along the chromosome.
    reference_genome : :py:class:`.hail.genetics.GenomeReference` (optional)
        Reference genome to use (uses :meth:`hail.api2.HailContext.default_reference` if not passed).

    Returns
    -------
    :py:class:`.hail.expr.expression.LocusExpression`
    """
    contig = to_expr(contig)
    pos = to_expr(pos)
    if reference_genome is None:
        reference_genome = Env.hc().default_reference
    indices, aggregations, joins = unify_all(contig, pos)
    return construct_expr(ApplyMethod('Locus({})'.format(reference_genome.name), contig._ast, pos._ast),
                          TLocus(reference_genome), indices, aggregations, joins)


@typecheck(s=expr_str, reference_genome=nullable(GenomeReference))
def parse_locus(s, reference_genome=None):
    """Construct a locus expression by parsing a string or string expression.

    Examples
    --------
    .. doctest::

        >>> eval_expr(functions.parse_locus("1:10000"))
        Locus(contig=1, position=10000, reference_genome=GRCh37)

    Notes
    -----
    This method expects strings of the form ``contig:position``, e.g. ``16:29500000``
    or ``X:123456``.

    Parameters
    ----------
    s : str or :py:class:`hail.expr.expression.StringExpression`
        String to parse.
    reference_genome : :py:class:`.hail.genetics.GenomeReference` (optional)
        Reference genome to use (uses :meth:`hail.api2.HailContext.default_reference` if not passed).

    Returns
    -------
    :py:class:`.hail.expr.expression.LocusExpression`
    """
    s = to_expr(s)
    if reference_genome is None:
        reference_genome = Env.hc().default_reference
    return construct_expr(ApplyMethod('Locus({})'.format(reference_genome.name), s._ast), TLocus(reference_genome),
                          s._indices, s._aggregations, s._joins)


@typecheck(start=expr_locus, end=expr_locus)
def interval(start, end):
    """Construct an interval expression from two loci.

    Examples
    --------
    .. doctest::

        >>> eval_expr(functions.interval(functions.locus("1", 100),
        ...                              functions.locus("1", 1000)))
        Interval(start=Locus(contig=1, position=100, reference_genome=GRCh37),
                 end=Locus(contig=1, position=1000, reference_genome=GRCh37))
    Parameters
    ----------
    start : :py:class:`.hail.genetics.Locus` or :py:class:`hail.expr.expression.LocusExpression`
        Starting locus (inclusive).
    end : :py:class:`.hail.genetics.Locus` or :py:class:`hail.expr.expression.LocusExpression`
        End locus (exclusive).
    reference_genome : :py:class:`.hail.genetics.GenomeReference` (optional)
        Reference genome to use (uses :meth:`hail.api2.HailContext.default_reference` if not passed).

    Returns
    -------
    :py:class:`.hail.expr.expression.IntervalExpression`
    """
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
    """Construct an interval expression by parsing a string or string expression.

    Examples
    --------
    .. doctest::

        >>> eval_expr(functions.parse_interval('1:1000-2000'))
        Interval(start=Locus(contig=1, position=1000, reference_genome=GRCh37),
                 end=Locus(contig=1, position=2000, reference_genome=GRCh37))

        >>> eval_expr(functions.parse_interval('1:start-10M'))
        Interval(start=Locus(contig=1, position=0, reference_genome=GRCh37),
                 end=Locus(contig=1, position=10000000, reference_genome=GRCh37))

    Notes
    -----
    This method expects strings of the form ``contig:start-end``, e.g. ``16:29500000-30200000``,
    ``8:start-end``, or ``X:10M-20M``.

    Parameters
    ----------
    s : str or :py:class:`hail.expr.expression.StringExpression`
        String to parse.
    reference_genome : :py:class:`.hail.genetics.GenomeReference` (optional)
        Reference genome to use (uses :meth:`hail.api2.HailContext.default_reference` if not passed).

    Returns
    -------
    :py:class:`.hail.expr.expression.IntervalExpression`
    """
    s = to_expr(s)
    if reference_genome is None:
        reference_genome = Env.hc().default_reference
    return construct_expr(
        ApplyMethod('Interval({})'.format(reference_genome.name), s._ast), TInterval(reference_genome),
        s._indices, s._aggregations, s._joins)


@typecheck(contig=expr_str, pos=expr_int32, ref=expr_str, alts=oneof(listof(expr_str), expr_list),
           reference_genome=nullable(GenomeReference))
def variant(contig, pos, ref, alts, reference_genome=None):
    """Construct a variant expression from fields.

    Examples
    --------
    .. doctest::

        >>> eval_expr(functions.variant('1', 1000, 'A', ['TC', 'T']))
        Variant(contig=1, start=1000, ref=A,
                alts=[AltAllele(ref=A, alt=TC), AltAllele(ref=A, alt=T)], reference_genome=GRCh37)

    Parameters
    ----------
    contig : str or :py:class:`hail.expr.expression.StringExpression`
        Chromosome.
    pos : int or :py:class:`hail.expr.expression.Int32Expression`
        Base position along the chromosome.
    ref : str or :py:class:`hail.expr.expression.StringExpression`
        Reference allele.
    alts : :py:class:`hail.expr.expression.ArrayExpression` or list of str or :py:class:`hail.expr.expression.StringExpression`
        List of alternate alleles.
    reference_genome : :py:class:`.hail.genetics.GenomeReference` (optional)
        Reference genome to use (uses :meth:`hail.api2.HailContext.default_reference` if not passed).

    Returns
    -------
    :py:class:`.hail.expr.expression.VariantExpression`
    """
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
    """Construct a variant expression by parsing a string or string expression.

    Examples
    --------
    .. doctest::

        >>> eval_expr(functions.parse_variant('1:1000:A:TC,T'))
        Variant(contig=1, start=1000, ref=A,
                alts=[AltAllele(ref=A, alt=TC), AltAllele(ref=A, alt=T)], reference_genome=GRCh37)

    Notes
    -----
    This method expects strings of the form ``chromosome:position:ref:alt1,alt2...``, like ``1:1:A:T``
    or ``1:100:A:TC,C``.

    Parameters
    ----------
    s : str or :py:class:`hail.expr.expression.StringExpression`
        String to parse.
    reference_genome : :py:class:`.hail.genetics.GenomeReference` (optional)
        Reference genome to use (uses :meth:`hail.api2.HailContext.default_reference` if not passed).

    Returns
    -------
    :py:class:`.hail.expr.expression.VariantExpression`
    """
    s = to_expr(s)
    if reference_genome is None:
        reference_genome = Env.hc().default_reference
    return construct_expr(ApplyMethod('Variant({})'.format(reference_genome.name), s._ast),
                          TVariant(reference_genome), s._indices, s._aggregations, s._joins)


@args_to_expr
def call(i):
    """Construct a call expression from an integer or integer expression.

    Examples
    --------
    .. doctest::

        >>> eval_expr(functions.call(1))
        Call(gt=1)

    Notes
    -----
    This method expects one argument, the triangular number of the two allele
    indices. In order to construct a call expression from two allele indices, first
    use :py:meth:`hail.expr.functions.gt_index`.

    Parameters
    ----------
    i : int or :py:class:`hail.expr.expressions.Int32Expression`
        Triangular number of new call.

    Returns
    -------
    :py:class:`hail.expr.expressions.CallExpression`
    """
    return CallExpression(ApplyMethod('Call', i._ast), TCall(), i._indices, i._aggregations, i._joins)


@args_to_expr
@typecheck(expression=anytype)
def is_defined(expression):
    """Returns ``True`` if the argument is not missing.

    Examples
    --------
    .. doctest::

        >>> eval_expr(functions.is_defined(5))
        True

        >>> eval_expr(functions.is_defined(functions.null(TString())))
        False

        >>> eval_expr(functions.is_defined(functions.null(TBoolean()) & True))
        False

    Parameters
    ----------
    expression
        Expression to test.

    Returns
    -------
    :py:class:`hail.expr.expressions.BooleanExpression`
        ``True`` if `expression` is not missing, ``False`` otherwise.
    """
    return _func("isDefined", TBoolean(), expression)


@args_to_expr
@typecheck(expression=anytype)
def is_missing(expression):
    """Returns ``True`` if the argument is missing.

    Examples
    --------
    .. doctest::

        >>> eval_expr(functions.is_missing(5))
        False

        >>> eval_expr(functions.is_missing(functions.null(TString())))
        True

        >>> eval_expr(functions.is_missing(functions.null(TBoolean()) & True))
        True

    Parameters
    ----------
    expression
        Expression to test.

    Returns
    -------
    :py:class:`hail.expr.expressions.BooleanExpression`
        ``True`` if `expression` is missing, ``False`` otherwise.
    """
    return _func("isMissing", TBoolean(), expression)


@args_to_expr
@typecheck(x=expr_numeric)
def is_nan(x):
    """Returns ``True`` if the argument is ``NaN`` (not a number).

    Examples
    --------
    .. doctest::

        >>> eval_expr(functions.is_nan(0))
        False

        >>> eval_expr(functions.is_nan(functions.capture(0) / 0))
        True

        >>> eval_expr(functions.is_nan(functions.capture(0) / functions.null(TFloat64())))
        None

    Notes
    -----
    Note that :py:meth:`is_missing` will return ``False`` on ``NaN`` since ``NaN``
    is a defined value. Additionally, this method will return missing if `x` is
    missing.

    Parameters
    ----------
    x : float or :py:class:`hail.expr.expressions.Float64Expression`
        Expression to test.

    Returns
    -------
    :py:class:`hail.expr.expressions.BooleanExpression`
        ``True`` if `x` is ``NaN``, ``False`` otherwise.
    """
    return _func("isnan", TBoolean(), x)


@args_to_expr
@typecheck(x=anytype)
def json(x):
    """Convert an expression to a JSON string expression.

    Examples
    --------
    .. doctest::

        >>> eval_expr(functions.json([1,2,3,4,5]))
        '[1,2,3,4,5]'

        >>> eval_expr(functions.json(Struct(a='Hello', b=0.12345, c=[1,2], d={'hi', 'bye'})))
        '{"a":"Hello","c":[1,2],"b":0.12345,"d":["bye","hi"]}'

    Parameters
    ----------
    x
        Expression to convert.

    Returns
    -------
    :py:class:`hail.expr.expressions.StringExpression`
        String expression with JSON representation of `x`.
    """
    return _func("json", TString(), x)


@typecheck(x=expr_numeric, base=nullable(expr_numeric))
def log(x, base=None):
    """Take the logarithm of the `x` with base `base`.

    Examples
    --------
    .. doctest::

        >>> eval_expr(functions.log(10))
        2.302585092994046

        >>> eval_expr(functions.log(10, 10))
        1.0

        >>> eval_expr(functions.log(1024, 2))
        10.0

    Notes
    -----
    If the `base` argument is not supplied, then the natural logarithm is used.

    Parameters
    ----------
    x : float or :py:class:`hail.expr.expressions.Float64Expression`
    base : float or :py:class:`hail.expr.expressions.Float64Expression`

    Returns
    -------
    :py:class:`hail.expr.expressions.Float64Expression`
    """
    x = to_expr(x)
    if base is not None:
        return _func("log", TFloat64(), x, to_expr(base))
    else:
        return _func("log", TFloat64(), x)


@args_to_expr
@typecheck(x=expr_numeric)
def log10(x):
    """Take the logarithm of the `x` with base 10.

    Examples
    --------
    .. doctest::

        >>> eval_expr(functions.log10(1000))
        3.0

        >>> eval_expr(functions.log10(0.0001123))
        -3.949620243738542

    Parameters
    ----------
    x : float or :py:class:`hail.expr.expressions.Float64Expression`

    Returns
    -------
    :py:class:`hail.expr.expressions.Float64Expression`
    """
    return _func("log10", TFloat64(), x)


@args_to_expr
@typecheck(b=expr_bool)
def logical_not(b):
    """Negates a boolean.

    Examples
    --------
    .. doctest::

        >>> eval_expr(functions.logical_not(False))
        True

        >>> eval_expr(functions.logical_not(True))
        False

        >>> eval_expr(functions.logical_not(functions.null(TBoolean())))
        None

    Notes
    -----
    Returns ``False`` when `b` is ``True``, returns ``True`` when `b` is ``False``.
    When `b` is missing, returns missing.

    Parameters
    ----------
    b : bool or :py:class:`.hail.expr.expressions.BooleanExpression`

    Returns
    -------
    :py:class:`.hail.expr.expressions.BooleanExpression`
    """
    return _func("!", TBoolean(), b)


@typecheck(s1=StructExpression, s2=StructExpression)
@args_to_expr
def merge(s1, s2):
    ret_type = s1._type._merge(s2._type)
    return _func("merge", ret_type, s1, s2)


@typecheck(a=anytype, b=anytype)
@args_to_expr
def or_else(a, b):
    """If `a` is missing, return `b`.

    Examples
    --------
    .. doctest::

        >>> eval_expr(functions.or_else(5, 7))
        5

        >>> eval_expr(functions.or_else(functions.null(TInt32()), 7))
        7

    Parameters
    ----------
    a
    b

    Returns
    -------
    :py:class:`hail.expr.expression.Expression`
    """
    a = to_expr(a)
    # FIXME: type promotion
    return _func("orElse", a._type, a, b)


@args_to_expr
@typecheck(predicate=expr_bool, value=anytype)
def or_missing(predicate, value):
    """Returns `value` if `predicate` is ``True``, otherwise returns missing.

    Examples
    --------
    .. doctest::

        >>> eval_expr(functions.or_missing(True, 5))
        5

        >>> eval_expr(functions.or_missing(False, 5))
        None

    Parameters
    ----------
    predicate : bool or :py:class:`hail.expr.expression.BooleanExpression`
    value : Value to return if `predicate` is true.

    Returns
    -------
    :py:class:`hail.expr.expression.Expression`
        This expression has the same type as `b`.
    """
    predicate = to_expr(predicate)
    return _func("orMissing", predicate._type, predicate, value)

@typecheck(x=expr_int32, n=expr_int32, p=expr_numeric,
           alternative=enumeration("two.sided", "greater", "less"))
@args_to_expr
def binom_test(x, n, p, alternative):
    """Performs a binomial test on `p` given `x` successes in `n` trials.

    Examples
    --------
    .. doctest::

        >>> eval_expr(functions.binom_test(5, 10, 0.5, 'less'))
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
    x : int or :class:`Int32Expression`
        Number of successes.
    n : int or :class:`Int32Expression`
        Number of trials.
    p : float or :class:`Float64Expression`
        Probability of success, between 0 and 1.
    alternative
        : One of, "two.sided", "greater", "less".

    Returns
    -------
    :class:`Float64Expression`
        p-value.
    """
    return _func("binomTest", TFloat64(), x, n, p, alternative)

@typecheck(x=expr_numeric, df=expr_numeric)
@args_to_expr
def pchisqtail(x, df):
    """Returns the probability under the right-tail starting at x for a chi-squared
    distribution with df degrees of freedom.

    Examples
    --------
    .. doctest::

        >>> eval_expr(functions.pchisqtail(5, 1))
        0.025347318677468304

    Parameters
    ----------
    x : float or :py:class:`hail.expr.expressions.Float64Expression`
    df : float or :py:class:`hail.expr.expressions.Float64Expression`
        Degrees of freedom.

    Returns
    -------
    :py:class:`hail.expr.expressions.Float64Expression`
    """
    return _func("pchisqtail", TFloat64(), x, df)


@typecheck(x=expr_numeric)
@args_to_expr
def pnorm(x):
    """The cumulative probability function of a standard normal distribution.

    Examples
    --------
    .. doctest::

        >>> eval_expr(functions.pnorm(0))
        0.5

        >>> eval_expr(functions.pnorm(1))
        0.8413447460685429

        >>> eval_expr(functions.pnorm(2))
        0.9772498680518208

    Notes
    -----
    Returns the left-tail probability `p` = Prob(:math:Z < x) with :math:Z a standard normal random variable.

    Parameters
    ----------
    x : float or :py:class:`hail.expr.expression.Float64Expression`

    Returns
    -------
    :py:class:`hail.expr.expression.Float64Expression`
    """
    return _func("pnorm", TFloat64(), x)


@typecheck(x=expr_numeric, lamb=expr_numeric, lower_tail=expr_bool, log_p=expr_bool)
@args_to_expr
def ppois(x, lamb, lower_tail=True, log_p=False):
    """The cumulative probability function of a Poisson distribution.

    Examples
    --------
    .. doctest::

        >>> eval_expr(functions.ppois(2, 1))
        0.9196986029286058

    Notes
    -----
    If `lower_tail` is true, returns Prob(:math:`X \leq` `x`) where :math:`X` is a
    Poisson random variable with rate parameter `lamb`. If `lower_tail` is false,
    returns Prob(:math:`X` > `x`).

    Parameters
    ----------
    x : float or :py:class:`hail.expr.expression.Float64Expression`
    lamb : float or :py:class:`hail.expr.expression.Float64Expression`
        Rate parameter of Poisson distribution.
    lower_tail : bool or :py:class:`hail.expr.expression.BooleanExpression`
        If ``True``, compute the probability of an outcome at or below `x`,
        otherwise greater than `x`.
    log_p : bool or :py:class:`hail.expr.expression.BooleanExpression`
        Return the natural logarithm of the probability.

    Returns
    -------
    :py:class:`hail.expr.expression.Float64Expression`
    """
    return _func("ppois", TFloat64(), x, lamb, lower_tail, log_p)


@typecheck(p=expr_numeric, df=expr_numeric)
@args_to_expr
def qchisqtail(p, df):
    """Inverts :py:meth:`hail.expr.functions.pchisqtail`.

    Examples
    --------
    .. doctest::

        >>> eval_expr(functions.qchisqtail(0.01, 1))
        6.634896601021213

    Notes
    -----
    Returns right-quantile `x` for which `p` = Prob(:math:`Z^2` > x) with :math:`Z^2` a chi-squared random
     variable with degrees of freedom specified by `df`. `p` must satisfy 0 < `p` <= 1.

    Parameters
    ----------
    p : float or :py:class:`hail.expr.expression.Float64Expression`
        Probability.
    df : float or :py:class:`hail.expr.expression.Float64Expression`
        Degrees of freedom.

    Returns
    -------
    :py:class:`hail.expr.expression.Float64Expression`
    """
    return _func("qchisqtail", TFloat64(), p, df)


@typecheck(p=expr_numeric)
@args_to_expr
def qnorm(p):
    """Inverts :py:meth:`hail.expr.functions.pnorm`.

    Examples
    --------
    .. doctest::

        >>> eval_expr(functions.qnorm(0.90))
        1.2815515655446008

    Notes
    -----
    Returns left-quantile `x` for which p = Prob(:math:`Z` < x) with :math:`Z` a standard normal random variable.
    `p` must satisfy 0 < `p` < 1.

    Parameters
    ----------
    p : float or :py:class:`hail.expr.expression.Float64Expression`
        Probability.

    Returns
    -------
    :py:class:`hail.expr.expression.Float64Expression`
    """
    return _func("qnorm", TFloat64(), p)


@typecheck(p=expr_numeric, lamb=expr_numeric, lower_tail=expr_bool, log_p=expr_bool)
@args_to_expr
def qpois(p, lamb, lower_tail=True, log_p=False):
    """Inverts :py:meth:`hail.expr.functions.ppois`.

    Examples
    --------
    .. doctest::

        >>> eval_expr(functions.qpois(0.99, 1))
        4

    Notes
    -----
    Returns the smallest integer :math:`x` such that Prob(:math:`X \leq x`) :math:`\geq` `p` where :math:`X`
    is a Poisson random variable with rate parameter `lambda`.

    Parameters
    ----------
    p : float or :py:class:`hail.expr.expression.Float64Expression`
    lamb : float or :py:class:`hail.expr.expression.Float64Expression`
        Rate parameter of Poisson distribution.
    lower_tail : bool or :py:class:`hail.expr.expression.BooleanExpression`
        Corresponds to `lower_tail` parameter in inverse :py:meth:`ppois`.
    log_p : bool or :py:class:`hail.expr.expression.BooleanExpression`
        Exponentiate `p` before testing.

    Returns
    -------
    :py:class:`hail.expr.expression.Float64Expression`
    """
    return _func("qpois", TInt32(), p, lamb, lower_tail, log_p)


@typecheck(start=expr_int32, stop=expr_int32, step=expr_int32)
@args_to_expr
def range(start, stop, step=1):
    """Returns an array of integers from `start` to `stop` by `step`.

    Examples
    --------
    .. doctest::

        >>> eval_expr(functions.range(0, 10))
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

        >>> eval_expr(functions.range(0, 10, step=3))
        [0, 3, 6, 9]

    Notes
    -----
    The range includes `start`, but excludes `stop`.

    Parameters
    ----------
    start : int or :py:class:`hail.expr.expression.Int32Expression`
        Start of range.
    stop : int or :py:class:`hail.expr.expression.Int32Expression`
        End of range.
    step : int or :py:class:`hail.expr.expression.Int32Expression`
        Step of range.

    Returns
    -------
    :py:class:`hail.expr.expression.ArrayInt32Expression`
    """
    return _func("range", TArray(TInt32()), start, stop, step)


@typecheck(p=expr_numeric)
@args_to_expr
def rand_bool(p):
    """Returns ``True`` with probability `p` (RNG).

    Examples
    --------
    .. doctest::

        >>> eval_expr(functions.rand_bool(0.5))
        True

        >>> eval_expr(functions.rand_bool(0.5))
        False

    Warning
    -------
    This function is non-deterministic, meaning that successive runs of the same pipeline including
    RNG expressions may return different results. This is a known bug.

    Parameters
    ----------
    p : float or :py:class:`hail.expr.expression.Float64Expression`
        Probability between 0 and 1.

    Returns
    -------
    :py:class:`hail.expr.expression.BooleanExpression`
    """
    return _func("pcoin", TBoolean(), p)


@typecheck(mean=expr_numeric, sd=expr_numeric)
@args_to_expr
def rand_norm(mean=0, sd=1):
    """Samples from a normal distribution with mean `mean` and standard deviation `sd` (RNG).

    Examples
    --------
    .. doctest::

        >>> eval_expr(functions.rand_norm())
        1.5388475315213386

        >>> eval_expr(functions.rand_norm())
        -0.3006188509144124

    Warning
    -------
    This function is non-deterministic, meaning that successive runs of the same
    pipeline including RNG expressions may return different results. This is a known
    bug.

    Parameters
    ----------
    mean : float or :py:class:`hail.expr.expression.Float64Expression`
        Mean of normal distribution.
    sd : float or :py:class:`hail.expr.expression.Float64Expression`
        Standard deviation of normal distribution.

    Returns
    -------
    :py:class:`hail.expr.expression.Float64Expression`
    """
    return _func("rnorm", TFloat64(), mean, sd)


@args_to_expr
@typecheck(lamb=expr_numeric)
def rand_pois(lamb):
    """Samples from a Poisson distribution with rate parameter `lamb` (RNG).

    Examples
    --------
    .. doctest::

        >>> eval_expr(functions.rand_pois(1))
        2.0

        >>> eval_expr(functions.rand_pois(1))
        3.0

    Warning
    -------
    This function is non-deterministic, meaning that successive runs of the same
    pipeline including RNG expressions may return different results. This is a known
    bug.

    Parameters
    ----------
    lamb : float or :py:class:`hail.expr.expression.Float64Expression`
        Rate parameter for Poisson distribution.

    Returns
    -------
    :py:class:`hail.expr.expression.Float64Expression`
    """
    return _func("rpois", TFloat64(), lamb)


@args_to_expr
@typecheck(min=expr_numeric, max=expr_numeric)
def rand_unif(min, max):
    """Returns a random floating-point number uniformly drawn from the interval [`min`, `max`].

    Examples
    --------
    .. doctest::

        >>> eval_expr(functions.rand_unif(0, 1))
        0.7983073825816226

        >>> eval_expr(functions.rand_unif(0, 1))
        0.5161799497741769

    Warning
    -------
    This function is non-deterministic, meaning that successive runs of the same
    pipeline including RNG expressions may return different results. This is a known
    bug.

    Parameters
    ----------
    min : float or :py:class:`hail.expr.expression.Float64Expression`
        Left boundary of range.
    max : float or :py:class:`hail.expr.expression.Float64Expression`
        Right boundary of range.

    Returns
    -------
    :py:class:`hail.expr.expression.Float64Expression`
    """
    return _func("runif", TFloat64(), min, max)


@typecheck(s=oneof(Struct, StructExpression), identifiers=tupleof(expr_str))
def select(s, *identifiers):
    s = to_expr(s)
    ret_type = s._type._select(*identifiers)
    return construct_expr(StructOp('select', s._ast, *identifiers), ret_type, s._indices, s._aggregations, s._joins)


@typecheck(x=expr_numeric)
@args_to_expr
def sqrt(x):
    """Returns the square root of `x`.

    Examples
    --------
    .. doctest::

        >>> eval_expr(functions.sqrt(3))
        1.7320508075688772

    Notes
    -----
    It is also possible to exponentiate expression with standard Python syntax,
    e.g. ``x ** 0.5``.

    Parameters
    ----------
    x : float or :py:class:`hail.expr.expression.Float64Expression`

    Returns
    -------
    :py:class:`hail.expr.expression.Float64Expression`
    """
    return _func("sqrt", TFloat64(), x)


@typecheck(x=anytype)
@args_to_expr
def to_str(x):
    """Returns the string representation of `x`.

    Examples
    --------
    .. doctest::

        >>> eval_expr(functions.to_str(Struct(a=5, b=7)))
        '{"a":5,"b":7}'

    Parameters
    ----------
    x

    Returns
    -------
    :py:class:`hail.expr.expression.StringExpression`
    """
    return _func("str", TString(), x)
