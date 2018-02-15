from hail.matrixtable import MatrixTable
from hail.table import Table
from hail.utils.java import Env, handle_py4j, jarray_to_list, joption
from hail.utils import wrap_to_list
from hail.typecheck.check import typecheck
from hail.expr.expression import *
from hail.expr.ast import Reference


@handle_py4j
@typecheck(vertices=Expression,
           i=Expression,
           j=Expression,
           tie_breaker=nullable(func_spec(2, expr_numeric)))
def maximal_independent_set(vertices, i, j, tie_breaker=None):
    """Filter a table to contain only those rows that belong in a near `maximal independent set`,
    where edges between the rows are given by a two-column table.

    .. _maximal independent set: https://en.wikipedia.org/wiki/Maximal_independent_set

    Examples
    --------

    Prune individuals from a dataset until no close relationships remain with
    respect to a PC-Relate measure of kinship.

    >>> pc_rel = hl.pc_relate(dataset, 2, 0.001)
    >>> pairs = pc_rel.filter(pc_rel['kin'] > 0.125).select('i', 'j')
    >>> individuals = pairs.annotate(id=[pairs.i,pairs.j]).select('id').explode('id')
    >>> result_table = hl.maximal_independent_set(individuals.id, pairs.i, pairs.j)

    Prune individuals from a dataset, preferring to keep cases over controls.

    >>> pc_rel = hl.pc_relate(dataset, 2, 0.001)
    >>> pairs = pc_rel.filter(pc_rel['kin'] > 0.125).select('i', 'j')
    >>> samples = dataset.cols()
    >>> pairs_with_case = pairs.select(
    ...     i=hl.Struct(id=pairs.i, is_case=samples[pairs.i].isCase),
    ...     j=hl.Struct(id=pairs.j, is_case=samples[pairs.j].isCase))
    >>> def tie_breaker(l, r):
    ...     return hl.cond(l.is_case & ~r.is_case, -1,
    ...                    hl.cond(~l.is_case & r.is_case, 1, 0))
    >>> individuals = (pairs_with_case.annotate(id_with_case=[pairs_with_case.i, pairs_with_case.j])
    ...     .select('id_with_case').explode('id_with_case'))
    >>> result_table = hl.maximal_independent_set(
    ...     individuals.id_with_case, pairs_with_case.i, pairs_with_case.j, tie_breaker)

    Notes
    -----

    The vertex set of the graph is implicitly all the values realized by `i`
    and `j` on the rows of this table. Each row of the table corresponds to an
    undirected edge between the vertices given by evaluating `i` and `j` on
    that row. An undirected edge may appear multiple times in the table and
    will not affect the output. Vertices with self-edges are removed as they
    are not independent of themselves. Vertices with no edges will be kept in
    the final filtered table.

    The expressions for `vertices`, `i`, and `j` must have the same type.
    The vertices should be stored in a single column of one table, while the edges
    are stored as columns i and j of another table.

    This method implements a greedy algorithm which iteratively removes a
    vertex of highest degree until the graph contains no edges. The greedy
    algorithm always returns an independent set, but the set may not always
    be perfectly maximal.

    `tie_breaker` is a Python function taking two arguments---say `l` and
    `r`---each of which is an :class:`Expression` of the same type as `i` and
    `j`. `tie_breaker` returns a :class:`NumericExpression`, which defines an
    ordering on nodes. A pair of nodes can be ordered in one of three ways, and
    `tie_breaker` must encode the relationship as follows:

     - if ``l < r`` then ``tie_breaker`` evaluates to some negative integer
     - if ``l == r`` then ``tie_breaker`` evaluates to 0
     - if ``l > r`` then ``tie_breaker`` evaluates to some positive integer

    For example, the usual ordering on the integers is defined by: ``l - r``.

    When multiple nodes have the same degree, this algorithm will order the
    nodes according to ``tie_breaker`` and remove the *largest* node.

    Parameters
    ----------
    vertices : :class:`.Expression`
        Expression representing the nodes of the table being filtered. Must be same type as i and j.
    i : :class:`.Expression`
        Expression to compute one endpoint of an edge.
    j : :class:`.Expression`
        Expression to compute another endpoint of an edge.
    tie_breaker : function
        Function used to order nodes with equal degree.

    Returns
    -------
    :class:`Table` filtered to the rows that belong in the approximate maximal independent set.
    """
    vertices_source = vertices._indices.source
    if not isinstance(vertices_source, Table):
        raise ValueError("Expects an expression of 'Table', found {}".format(
            "expression of '{}'".format(
                vertices_source.__class__) if vertices_source is not None else 'scalar expression'))
    if i.dtype != j.dtype or vertices.dtype != i.dtype:
        raise ValueError("Expects arguments `vertices`, `i` and `j` to have same type. "
                         "Found {}, {} and {}.".format(vertices.dtype, i.dtype, j.dtype))
    source = i._indices.source
    if not isinstance(source, Table):
        raise ValueError("Expects an expression of 'Table', found {}".format(
            "expression of '{}'".format(
                source.__class__) if source is not None else 'scalar expression'))
    if i._indices.source != j._indices.source:
        raise ValueError(
            "Expects arguments `i` and `j` to be expressions of the same Table. "
            "Found\n{}\n{}".format(i, j))

    node_t = i.dtype
    l = construct_expr(Reference('l'), node_t)
    r = construct_expr(Reference('r'), node_t)
    if tie_breaker:
        tie_breaker_expr = tie_breaker(l, r).to_int64()
        edges, _ = source._process_joins(i, j, tie_breaker_expr)
        tie_breaker_hql = tie_breaker_expr._ast.to_hql()
    else:
        edges, _ = source._process_joins(i, j)
        tie_breaker_hql = None
    return Table(edges._jt.maximalIndependentSet(
        vertices_source._jt, vertices._ast.to_hql(), i._ast.to_hql(), j._ast.to_hql(), joption(tie_breaker_hql)))


def require_variant(dataset, method):
    if (dataset.row_key != ['locus', 'alleles'] or
            not isinstance(dataset['locus'].dtype, TLocus) or
            not dataset['alleles'].dtype == tarray(tstr)):
        raise TypeError("Method '{}' requires row keys 'locus' (Locus) and 'alleles' (Array[String])\n"
                        "  Found:{}".format(method, ''.join(
            "\n    '{}': {}".format(k, str(dataset[k].dtype)) for k in dataset.row_key)))


def require_locus(dataset, method):
    if (len(dataset.partition_key) != 1 or
            not isinstance(dataset[dataset.partition_key[0]].dtype, TLocus)):
        raise TypeError("Method '{}' requires partition key of type Locus.\n"
                        "  Found:{}".format(method, ''.join(
            "\n    '{}': {}".format(k, str(dataset[k].dtype)) for k in dataset.partition_key)))


@handle_py4j
@typecheck(dataset=MatrixTable, method=str)
def require_biallelic(dataset, method):
    require_variant(dataset, method)
    dataset = MatrixTable(Env.hail().methods.VerifyBiallelic.apply(dataset._jvds, method))
    return dataset


@handle_py4j
@typecheck(dataset=MatrixTable, name=str)
def rename_duplicates(dataset, name='unique_id'):
    """Rename duplicate column keys.

    .. include:: ../_templates/req_tstring.rst

    Examples
    --------

    >>> renamed = hl.rename_duplicates(dataset).cols()
    >>> duplicate_samples = (renamed.filter(renamed.s != renamed.unique_id)
    ...                             .select('s')
    ...                             .collect())

    Notes
    -----

    This method produces a new column field from the string column key by
    appending a unique suffix ``_N`` as necessary. For example, if the column
    key "NA12878" appears three times in the dataset, the first will produce
    "NA12878", the second will produce "NA12878_1", and the third will produce
    "NA12878_2". The name of this new field is parameterized by `name`.

    Parameters
    ----------
    dataset : :class:`.MatrixTable`
        Dataset.
    name : :obj:`str`
        Name of new field.

    Returns
    -------
    :class:`.MatrixTable`
    """

    return MatrixTable(dataset._jvds.renameDuplicates(name))


@handle_py4j
@typecheck(ds=MatrixTable,
           intervals=oneof(Interval, listof(Interval)),
           keep=bool)
def filter_intervals(ds, intervals, keep=True):
    """Filter rows with an interval or list of intervals.

    .. note::

        Requires the dataset to have a single partition key of type
        :class:`.TLocus`.

    Examples
    --------

    Filter to loci falling within one interval:

    >>> ds_result = hl.filter_intervals(dataset, hl.Interval.parse('17:38449840-38530994'))

    Remove all loci within list of intervals:

    >>> intervals = [hl.Interval.parse(x) for x in ['1:50M-75M', '2:START-400000', '3-22']]
    >>> ds_result = hl.filter_intervals(dataset, intervals)

    Notes
    -----
    This method takes an argument of :class:`.Interval` or list of
    :class:`.Interval`.

    Based on the ``keep`` argument, this method will either restrict to loci in
    the supplied interval ranges, or remove all loci in those ranges.  Note that
    intervals are left-inclusive, and right-exclusive; the below interval
    includes the locus ``15:100000`` but not ``15:101000``.

    >>> interval = hl.Interval.parse('15:100000-101000')

    When ``keep=True``, partitions that don't overlap any supplied interval
    will not be loaded at all.  This enables :func:`.filter_intervals` to be
    used for reasonably low-latency queries of small ranges of the genome, even
    on large datasets.

    Parameters
    ----------
    intervals : :class:`.Interval` or :obj:`list` of :class:`.Interval`
        Interval(s) to filter on.
    keep : :obj:`bool`
        If ``True``, keep only loci fall within any interval in `intervals`. If
        ``False``, keep only loci that fall outside all intervals in
        `intervals`.

    Returns
    -------
    :class:`.MatrixTable`
    """

    require_locus(ds, 'filter_intervals')

    intervals = wrap_to_list(intervals)
    jmt = Env.hail().methods.FilterIntervals.apply(ds._jvds, [x._jrep for x in intervals], keep)
    return MatrixTable(jmt)
