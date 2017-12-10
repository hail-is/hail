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


@decorator
def args_to_expr(func, *args):
    return func(*(to_expr(a) for a in args))


@typecheck(t=Type)
def null(t):
    """Creates a missing expression of a specified type.

    Since missingness cannot be automatically interpreted with :py:meth:`hail.expr.functions.broadcast`
    or :py:meth:`hail.expr.functions.capture`, this method is useful for constructing an expression that
    includes missing values.

    Parameters
    ----------
    t : :py:class:`hail.expr.Type`
        Type of the missing expression.

    Returns
    -------
    :py:class:`hail.expr.expression.Expression`
        A missing expression of type `t`.
    """
    return Expression(Literal('NA: {}'.format(t)), t)


def capture(x):
<<<<<<< 1f7f8e3e0988d9920bfe2a7eebd9071f5cc441d2
    return to_expr(x)
=======
    """Captures a Python variable or object as an expression.

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
    return convert_expr(x)
>>>>>>> Copy function docs to functions.py


def broadcast(x):
    """Broadcasts a Python variable or object as an expression.

    Good for using large Python objects in expressions. Using this function is equivalent
    to annotating the object as a global field of a :class:`hail.api2.Table` or
    :class:`hail.api2.MatrixTable` and using that global field in the expression.

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
            return Table(obj._hc, obj._jkt.annotateGlobalExpr('{} = {}'.format(uid, expr._ast.to_hql())))
        else:
            assert isinstance(obj, MatrixTable)
            return MatrixTable(obj._hc, obj._jvds.annotateGlobalExpr('global.{} = {}'.format(uid, expr._ast.to_hql())))

    return construct_expr(GlobalJoinReference(uid), expr._type, joins=(Join(joiner, [uid]),))


@typecheck(predicate=expr_bool, then_case=anytype, else_case=anytype)
@args_to_expr
def cond(predicate, then_case, else_case):
    """The Hail expression if/else statement; tests a condition and returns two options based on the result.

    If `predicate` evaluates to ``True``, returns `then_case`. If `predicate`
    evaluates to ``False``, returns `else_case`. If `predicate` is missing, then the
    result of the expression is missing.

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


@args_to_expr
@typecheck(c1=expr_int32, c2=expr_int32, c3=expr_int32, c4=expr_int32)
def chisq(c1, c2, c3, c4):
    """Calculates p-value (Chi-square approximation) and odds ratio for a 2x2 table.

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
    """Creates a dictionary from a list of keys and values.

    The length of `keys` and `values` must be identical, and all elements of each
    must be the same type.

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
    """Returns Prob(:math:`X` = `x`) from a Poisson distribution with rate parameter `lambda`.

    Parameters
    ----------
    x : float or :py:class:`hail.expr.expression.Float64Expression`
        Non-negative number at which to compute the probability density.
    lamb : float or :py:class:`hail.expr.expression.Float64Expression`
        Poisson rate parameter. Must be non-negative.
    log_p : bool or :py:class:`hail.expr.expression.BooleanExpression`
        If true, probabilities are returned as log(p).

    Returns
    -------
    :py:class:`hail.expr.expression.Float64Expression`
        A p-value or log p-value.
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
    """Computes `e` raised to the power of `x`.

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
    """Compute Hardy Weinberg Equilbrium p-value and heterozygosity ratio.

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

    Parameters
    ----------
    contig : str or :py:class:`hail.expr.expression.StringExpression`
        Chromosome.
    pos : int or :py:class:`hail.expr.expression.Int32Expression`
        Base position along the chromosome.
    reference_genome : :py:class:`.hail.genetics.GenomeReference` (optional)
        Reference genome to use (uses default reference if not passed).

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

    This method expects strings of the form ``contig:position``, like ``16:29500000``.

    Parameters
    ----------
    s : str or :py:class:`hail.expr.expression.StringExpression`
        String to parse.
    reference_genome : :py:class:`.hail.genetics.GenomeReference` (optional)
        Reference genome to use (uses default reference if not passed).

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

    Parameters
    ----------
    start : :py:class:`.hail.genetics.Locus` or :py:class:`hail.expr.expression.LocusExpression`
        Starting locus (inclusive).
    end : :py:class:`.hail.genetics.Locus` or :py:class:`hail.expr.expression.LocusExpression`
        End locus (exclusive).
    reference_genome : :py:class:`.hail.genetics.GenomeReference` (optional)
        Reference genome to use (uses default reference if not passed).

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

    This method expects strings of the form ``contig:start-end``, like ``16:29500000-30200000``,
    ``8:start-end``, or ``X:10M-20M``.

    Parameters
    ----------
    s : str or :py:class:`hail.expr.expression.StringExpression`
        String to parse.
    reference_genome : :py:class:`.hail.genetics.GenomeReference` (optional)
        Reference genome to use (uses default reference if not passed).

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

    Parameters
    ----------
    contig : str or :py:class:`hail.expr.expression.StringExpression`
        Chromosome.
    pos : int or :py:class:`hail.expr.expression.Int32Expression`
        Base position along the chromosome.
    ref : str or :py:class:`hail.expr.expression.StringExpression`
        Reference allele.
    alts : :py:class:`hail.expr.expression.ArrayExpression` or list of str or :py:class:`hail.expr.expression.StringExpression`
        Alternate allele(s).
    reference_genome : :py:class:`.hail.genetics.GenomeReference` (optional)
        Reference genome to use (uses default reference if not passed).

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

    This method expects strings of the form ``chromosome:position:ref:alt1,alt2...``, like ``1:1:A:T``
    or ``1:100:A:T,C``.

    Parameters
    ----------
    s : str or :py:class:`hail.expr.expression.StringExpression`
        String to parse.
    reference_genome : :py:class:`.hail.genetics.GenomeReference` (optional)
        Reference genome to use (uses default reference if not passed).

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

    This method expects one argument, the triangle number of the two alleles. In order
    to construct a call expression from two alleles, first use :py:meth:`hail.expr.functions.gt_index`.

    Parameters
    ----------
    i : int or :py:class:`hail.expr.expressions.Int32Expression`
        Triangle number of two alleles for new call.

    Returns
    -------
    :py:class:`hail.expr.expressions.CallExpression`
    """
    return CallExpression(ApplyMethod('Call', i._ast), TCall(), i._indices, i._aggregations, i._joins)


@args_to_expr
@typecheck(expression=anytype)
def is_defined(expression):
    """Returns true if the argument is not missing.

    Parameters
    ----------
    expression
        Expression for missingness test.

    Returns
    -------
    :py:class:`hail.expr.expressions.BooleanExpression`
        ``True`` if `expression` is not missing, ``False`` otherwise.
    """
    return _func("isDefined", TBoolean(), expression)


@args_to_expr
@typecheck(expression=anytype)
def is_missing(expression):
    """Returns true if the argument is missing.

    Parameters
    ----------
    expression
        Expression for missingness test.

    Returns
    -------
    :py:class:`hail.expr.expressions.BooleanExpression`
        ``True`` if `expression` is missing, ``False`` otherwise.
    """
    return _func("isMissing", TBoolean(), expression)


@args_to_expr
@typecheck(x=expr_numeric)
def is_nan(x):
    """Returns true if the argument is ``NaN`` (not a number).

    This method is different from :py:meth:`hail.expr.functions.is_missing`.

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

    Parameters
    ----------
    x
        Expression to convert to a JSON string.

    Returns
    -------
    :py:class:`hail.expr.expressions.StringExpression`
        String expression with JSON representation of `x`.
    """
    return _func("json", TString(), x)


@typecheck(x=expr_numeric, base=expr_numeric)
def log(x, base=None):
    """Take the logarithm of the `x` to base `base`.

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
    """Take the logarithm of the `x` to base 10.

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
    """Negates a boolean expression.

    Applied to a `True` expression, the result is `False`. Applied to a `False` expression, the
    result is `True`. Applied to a missing expression, the result is missing.

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
    """If `predicate` is true, return `b`. Otherwise, return missing.

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


@typecheck(x=expr_numeric, df=expr_numeric)
@args_to_expr
def pchisqtail(x, df):
    """Returns right-tail probability for `x` with `df` degrees of freedom.

    Returns right-tail probability p for which p = Prob(:math:`Z^2` > x) with :math:`Z^2` a chi-squared
    random variable with degrees of freedom specified by ``df``.

    Parameters
    ----------
    x : float or :py:class:`hail.expr.expressions.Float64Expression`
        Chi-squared statistic.
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
    """Returns left-tail probability p for which p = Prob(:math:`Z` < ``x``) with :math:`Z` a standard normal random variable.

    Parameters
    ----------
    x : float or :py:class:`hail.expr.expression.Float64Expression`

    Returns
    -------
    :py:class:`hail.expr.expression.Float64Expression`
        Left-tail p-value.
    """
    return _func("pnorm", TFloat64(), x)


@typecheck(x=expr_numeric, lamb=expr_numeric, lower_tail=expr_bool, log_p=expr_bool)
@args_to_expr
def ppois(x, lamb, lower_tail=True, log_p=False):
    """Computes tailed p-value for drawing `x` from a Poisson distribution.

    If `lower_tail` is true, returns Prob(:math:`X \leq` `x`) where :math:`X` is a Poisson random variable
    with rate parameter `lamb`. If `lowerTail` is false, returns Prob(:math:`X` > `x`).

    Parameters
    ----------
    x : float or :py:class:`hail.expr.expression.Float64Expression`
    lamb : float or :py:class:`hail.expr.expression.Float64Expression`
        Rate parameter of Poisson distribution.
    lower_tail : bool or :py:class:`hail.expr.expression.BooleanExpression`
        Test against the lower tail of the distribution.
    log_p : bool or :py:class:`hail.expr.expression.BooleanExpression`
        Return the natural logarithm of the p-value.

    Returns
    -------
    :py:class:`hail.expr.expression.Float64Expression`
    """
    return _func("ppois", TFloat64(), x, lamb, lower_tail, log_p)


@typecheck(p=expr_numeric, df=expr_numeric)
@args_to_expr
def qchisqtail(p, df):
    """Inverts :py:meth:`hail.expr.functions.pchisqtail`.

    Returns right-quantile `x` for which `p` = Prob(:math:`Z^2` > x) with :math:`Z^2` a chi-squared random
     variable with degrees of freedom specified by `df`. `p` must satisfy 0 < `p` <= 1.

    Parameters
    ----------
    p : float or :py:class:`hail.expr.expression.Float64Expression`
        p-value.
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

    Returns left-quantile `x` for which p = Prob(:math:`Z` < x) with :math:`Z` a standard normal random variable.
    `p` must satisfy 0 < `p` < 1.

    Parameters
    ----------
    p : float or :py:class:`hail.expr.expression.Float64Expression`
        p-value.

    Returns
    -------
    :py:class:`hail.expr.expression.Float64Expression`
    """
    return _func("qnorm", TFloat64(), p)


@typecheck(p=expr_numeric, lamb=expr_numeric, lower_tail=expr_bool, log_p=expr_bool)
@args_to_expr
def qpois(p, lamb, lower_tail=True, log_p=False):
    """Inverts :py:meth:`hail.expr.functions.ppois`.

    Returns the smallest integer :math:`x` such that Prob(:math:`X \leq x`) :math:`\geq` `p` where :math:`X`
    is a Poisson random variable with rate parameter `lambda`.

    Parameters
    ----------
    p : float or :py:class:`hail.expr.expression.Float64Expression`
    lamb : float or :py:class:`hail.expr.expression.Float64Expression`
        Rate parameter of Poisson distribution.
    lower_tail : bool or :py:class:`hail.expr.expression.BooleanExpression`
        Test against the lower tail of the distribution.
    log_p : bool or :py:class:`hail.expr.expression.BooleanExpression`
        Exponentiate `p` before testing.

    Returns
    -------
    :py:class:`hail.expr.expression.Float64Expression`
    """
    return _func("qpois", TInt32(), p, lamb, lower_tail, log_p)


@typecheck(stop=expr_int32, start=expr_int32, step=expr_int32)
@args_to_expr
def range(stop, start=0, step=1):
    """Returns an array of integers from `start` to `stop` by `step`.

    Parameters
    ----------
    stop : int or :py:class:`hail.expr.expression.Int32Expression`
        End of range.
    start : int or :py:class:`hail.expr.expression.Int32Expression`
        Start of range.
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
    """Return `True` with probability `p` (random number generator).

    Warning
    -------
    This function is non-deterministic, meaning that successive runs of the same pipeline including
    RNG expressions may return different results. This is a known bug, but is difficult to fix.

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
    """Returns a random sample from a normal distribution with mean `mean` and standard deviation `sd`.

    Warning
    -------
    This function is non-deterministic, meaning that successive runs of the same pipeline including
    RNG expressions may return different results. This is a known bug, but is difficult to fix.

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


@typecheck(lamb=expr_numeric)
def rand_pois(lamb):
    """Returns a random draw from a Poisson distribution with rate parameter `lamb`.

    Warning
    -------
    This function is non-deterministic, meaning that successive runs of the same pipeline including
    RNG expressions may return different results. This is a known bug, but is difficult to fix.

    Parameters
    ----------
    lamb : float or :py:class:`hail.expr.expression.Float64Expression`
        Rate parameter for Poisson distribution.

    Returns
    -------
    :py:class:`hail.expr.expression.Float64Expression`
    """
    return _func("rpois", TFloat64(), lamb)


@typecheck(min=expr_numeric, max=expr_numeric)
@args_to_expr
def rand_unif(min, max):
    """Returns a random floating-point number uniformly drawn from the range (`min`, `max`).

    Warning
    -------
    This function is non-deterministic, meaning that successive runs of the same pipeline including
    RNG expressions may return different results. This is a known bug, but is difficult to fix.

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

    Parameters
    ----------
    x

    Returns
    -------
    :py:class:`hail.expr.expression.StringExpression`
    """
    return _func("str", TString(), x)


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
