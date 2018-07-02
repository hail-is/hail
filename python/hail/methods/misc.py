import numpy as np

import hail as hl
from hail.expr.expr_ast import VariableReference
from hail.expr.expressions import *
from hail.expr.types import *
from hail.matrixtable import MatrixTable
from hail.table import Table
from hail.typecheck import *
from hail.utils import Interval, Struct
from hail.utils.java import Env, joption


@typecheck(i=Expression,
           j=Expression,
           keep=bool,
           tie_breaker=nullable(func_spec(2, expr_numeric)))
def maximal_independent_set(i, j, keep=True, tie_breaker=None) -> Table:
    """Return a table containing the vertices in a near
    `maximal independent set <https://en.wikipedia.org/wiki/Maximal_independent_set>`_
    of an undirected graph whose edges are given by a two-column table.

    Examples
    --------
    Run PC-relate and compute pairs of closely related individuals:

    >>> pc_rel = hl.pc_relate(dataset.GT, 0.001, k=2, statistics='kin')
    >>> pairs = pc_rel.filter(pc_rel['kin'] > 0.125)

    Starting from the above pairs, prune individuals from a dataset until no
    close relationships remain:

    >>> related_samples_to_remove = hl.maximal_independent_set(pairs.i, pairs.j, False)
    >>> result = dataset.filter_cols(
    ...     hl.is_defined(related_samples_to_remove[dataset.col_key]), keep=False)

    Starting from the above pairs, prune individuals from a dataset until no
    close relationships remain, preferring to keep cases over controls:

    >>> samples = dataset.cols()
    >>> pairs_with_case = pairs.key_by(
    ...     i=hl.struct(id=pairs.i, is_case=samples[pairs.i].is_case),
    ...     j=hl.struct(id=pairs.j, is_case=samples[pairs.j].is_case))
    >>> def tie_breaker(l, r):
    ...     return hl.cond(l.is_case & ~r.is_case, -1,
    ...                    hl.cond(~l.is_case & r.is_case, 1, 0))
    >>> related_samples_to_remove = hl.maximal_independent_set(
    ...    pairs_with_case.i, pairs_with_case.j, False, tie_breaker)
    >>> result = dataset.filter_cols(hl.is_defined(
    ...     related_samples_to_remove.key_by(
    ...        s = related_samples_to_remove.node.id.s)[dataset.col_key]), keep=False)

    Notes
    -----

    The vertex set of the graph is implicitly all the values realized by `i`
    and `j` on the rows of this table. Each row of the table corresponds to an
    undirected edge between the vertices given by evaluating `i` and `j` on
    that row. An undirected edge may appear multiple times in the table and
    will not affect the output. Vertices with self-edges are removed as they
    are not independent of themselves.

    The expressions for `i` and `j` must have the same type.

    The value of `keep` determines whether the vertices returned are those
    in the maximal independent set, or those in the complement of this set.
    This is useful if you need to filter a table without removing vertices that
    don't appear in the graph at all.

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

    The `tie_breaker` function must satisfy the following property:
    ``tie_breaker(l, r) == -tie_breaker(r, l)``.

    When multiple nodes have the same degree, this algorithm will order the
    nodes according to ``tie_breaker`` and remove the *largest* node.

    Parameters
    ----------
    i : :class:`.Expression`
        Expression to compute one endpoint of an edge.
    j : :class:`.Expression`
        Expression to compute another endpoint of an edge.
    keep : :obj:`bool`
        If ``True``, return vertices in set. If ``False``, return vertices removed.
    tie_breaker : function
        Function used to order nodes with equal degree.

    Returns
    -------
    :class:`.Table`
        Table with the set of independent vertices. The table schema is one row
        field `node` which has the same type as input expressions `i` and `j`.
    """

    if i.dtype != j.dtype:
        raise ValueError("'maximal_independent_set' expects arguments `i` and `j` to have same type. "
                         "Found {} and {}.".format(i.dtype, j.dtype))

    source = i._indices.source
    if not isinstance(source, Table):
        raise ValueError("'maximal_independent_set' expects an expression of 'Table'. Found {}".format(
            "expression of '{}'".format(
                source.__class__) if source is not None else 'scalar expression'))

    if i._indices.source != j._indices.source:
        raise ValueError(
            "'maximal_independent_set' expects arguments `i` and `j` to be expressions of the same Table. "
            "Found\n{}\n{}".format(i, j))

    node_t = i.dtype

    if tie_breaker:
        wrapped_node_t = ttuple(node_t)
        l = construct_expr(VariableReference('l'), wrapped_node_t)
        r = construct_expr(VariableReference('r'), wrapped_node_t)
        tie_breaker_expr = hl.int64(tie_breaker(l[0], r[0]))
        t, _ = source._process_joins(i, j, tie_breaker_expr)
        tie_breaker_hql = tie_breaker_expr._ast.to_hql()
    else:
        t, _ = source._process_joins(i, j)
        tie_breaker_hql = None

    nodes = (t.select(node=[i, j])
             .explode('node')
             .key_by('node')
             .select())

    edges = t.key_by(None).select('i', 'j')
    nodes_in_set = Env.hail().utils.Graph.maximalIndependentSet(edges._jt.collect(), node_t._jtype, joption(tie_breaker_hql))

    nt = Table(nodes._jt.annotateGlobal(nodes_in_set, hl.tset(node_t)._jtype, 'nodes_in_set'))
    nt = (nt
          .filter(nt.nodes_in_set.contains(nt.node), keep)
          .drop('nodes_in_set'))

    return nt


def require_col_key_str(dataset: MatrixTable, method: str):
    if not len(dataset.col_key) == 1 or dataset[next(iter(dataset.col_key))].dtype != hl.tstr:
        raise ValueError(f"Method '{method}' requires column key to be one field of type 'str', found "
                         f"{list(str(x.dtype) for x in dataset.col_key.values())}")


def require_row_key_variant(dataset, method):
    if (list(dataset.row_key) != ['locus', 'alleles'] or
            not isinstance(dataset['locus'].dtype, tlocus) or
            not dataset['alleles'].dtype == tarray(tstr)):
        raise ValueError("Method '{}' requires row key to be two fields 'locus' (type 'locus<any>') and "
                         "'alleles' (type 'array<str>')\n"
                         "  Found:{}".format(method, ''.join(
            "\n    '{}': {}".format(k, str(dataset[k].dtype)) for k in dataset.row_key)))


def require_row_key_variant_w_struct_locus(dataset, method):
    if (list(dataset.row_key) != ['locus', 'alleles'] or
            not dataset['alleles'].dtype == tarray(tstr) or
            (not isinstance(dataset['locus'].dtype, tlocus) and
                     dataset['locus'].dtype != hl.dtype('struct{contig: str, position: int32}'))):
        raise ValueError("Method '{}' requires row key to be two fields 'locus'"
                         " (type 'locus<any>' or 'struct{contig: str, position: int32}') and "
                         "'alleles' (type 'array<str>')\n"
                         "  Found:{}".format(method, ''.join(
            "\n    '{}': {}".format(k, str(dataset[k].dtype)) for k in dataset.row_key)))


def require_partition_key_locus(dataset, method):
    if (len(dataset.partition_key) != 1 or
            not isinstance(dataset.partition_key[0].dtype, tlocus)):
        raise ValueError("Method '{}' requires partition key to be one field of type 'locus<any>'.\n"
                         "  Found:{}".format(method, ''.join(
            "\n    '{}': {}".format(k, str(dataset[k].dtype)) for k in dataset.partition_key)))


def require_first_key_field_locus(table, method):
    if (len(table.key) == 0 or
            not isinstance(table.key[0].dtype, tlocus)):
        raise ValueError("Method '{}' requires first key field of type 'locus<any>'.\n"
                         "  Found:{}".format(method, ''.join(
            "\n    '{}': {}".format(k, str(table[k].dtype)) for k in table.key)))


@typecheck(table=Table, method=str)
def require_key(table, method):
    if table.key is None:
        raise ValueError("Method '{}' requires keyed table".format(method))


@typecheck(dataset=MatrixTable, method=str)
def require_biallelic(dataset, method) -> MatrixTable:
    require_row_key_variant(dataset, method)
    dataset = MatrixTable(Env.hail().methods.VerifyBiallelic.apply(dataset._jvds, method))
    return dataset


@typecheck(dataset=MatrixTable, name=str)
def rename_duplicates(dataset, name='unique_id') -> MatrixTable:
    """Rename duplicate column keys.

    .. include:: ../_templates/req_tstring.rst

    Examples
    --------

    >>> renamed = hl.rename_duplicates(dataset).cols()
    >>> duplicate_samples = (renamed.filter(renamed.s != renamed.unique_id)
    ...                             .select()
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


@typecheck(ds=MatrixTable,
           intervals=expr_array(expr_interval(expr_any)),
           keep=bool)
def filter_intervals(ds, intervals, keep=True) -> MatrixTable:
    """Filter rows with a list of intervals.

    Examples
    --------

    Filter to loci falling within one interval:

    >>> ds_result = hl.filter_intervals(dataset, [hl.parse_locus_interval('17:38449840-38530994')])

    Remove all loci within list of intervals:

    >>> intervals = [hl.parse_locus_interval(x) for x in ['1:50M-75M', '2:START-400000', '3-22']]
    >>> ds_result = hl.filter_intervals(dataset, intervals)

    Notes
    -----
    Based on the ``keep`` argument, this method will either restrict to points
    in the supplied interval ranges, or remove all rows in those ranges.

    When ``keep=True``, partitions that don't overlap any supplied interval
    will not be loaded at all.  This enables :func:`.filter_intervals` to be
    used for reasonably low-latency queries of small ranges of the dataset, even
    on large datasets.

    Parameters
    ----------
    ds : :class:`.MatrixTable`
        Dataset.
    intervals : :class:`.ArrayExpression` of type :py:data:`.tinterval`
        Intervals to filter on. If there is only one row partition key, the
        point type of the interval can be the type of the first partition key.
        Otherwise, the interval point type must be a :class:`.Struct` matching
        the row partition key schema.
    keep : :obj:`bool`
        If ``True``, keep only rows that fall within any interval in `intervals`.
        If ``False``, keep only rows that fall outside all intervals in
        `intervals`.

    Returns
    -------
    :class:`.MatrixTable`
    """

    n_pk = len(ds.partition_key)
    pk_type = ds.partition_key.dtype
    point_type = intervals.dtype.element_type.point_type

    if point_type == pk_type:
        needs_wrapper = False
    elif n_pk == 1 and point_type == ds.partition_key[0].dtype:
        needs_wrapper = True
    else:
        raise TypeError("The point type does not match the row partition key type of the dataset ('{}', '{}')".format(repr(point_type), repr(pk_type)))

    def wrap_input(interval):
        if interval is None:
            raise TypeError("'filter_intervals' does not allow missing values in 'intervals'.")
        elif needs_wrapper:
            return Interval(Struct(foo=interval.start),
                            Struct(foo=interval.end),
                            interval.includes_start,
                            interval.includes_end)
        else:
            return interval

    intervals = [wrap_input(x)._jrep for x in intervals.value]
    jmt = Env.hail().methods.FilterIntervals.apply(ds._jvds, intervals, keep)
    return MatrixTable(jmt)

@typecheck(mt=MatrixTable, bp_window_size=int)
def window_by_locus(mt: MatrixTable, bp_window_size: int) -> MatrixTable:
    """Collect arrays of row and entry values from preceding loci.

    .. include:: ../_templates/req_tlocus.rst

    .. include:: ../_templates/experimental.rst

    Examples
    --------
    >>> ds_result = hl.window_by_locus(ds, 3)

    Notes
    -----
    This method groups each row (variant) with the previous rows in a window of
    `bp_window_size` base pairs, putting the row values from the previous
    variants into `prev_rows` (row field of type ``array<struct>``) and entry
    values from those variants into `prev_entries` (entry field of type
    ``array<struct>``).

    The `bp_window_size` argument is inclusive; if `base_pairs` is 2 and the
    loci are

    .. code-block:: text

        1:100
        1:100
        1:102
        1:102
        1:103
        2:100
        2:101

    then the size of `prev_rows` is 0, 1, 2, 3, 2, 0, and 1, respectively (and
    same for the size of prev_entries).

    Parameters
    ----------
    mt : :class:`.MatrixTable`
        Input dataset.
    bp_window_size : :obj:`int`
        Base pairs to include in the backwards window (inclusive).

    Returns
    -------
    :class:`.MatrixTable`
    """
    require_partition_key_locus(mt, 'window_by_locus')
    return MatrixTable(mt._jvds.windowVariants(bp_window_size))


@typecheck(a=np.ndarray,
           radius=oneof(int, float))
def array_windows(a, radius):
    """Returns start and stop indices for window around each array value.

    Examples
    --------

    >>> hl.array_windows(np.array([1, 2, 4, 4, 6, 8]), 2)
    (array([0, 0, 1, 1, 2, 4]), array([2, 4, 5, 5, 6, 6]))

    >>> hl.array_windows(np.array([-10.0, -2.5, 0.0, 0.0, 1.2, 2.3, 3.0]), 2.5)
    (array([0, 1, 1, 1, 2, 2, 4]), array([1, 4, 6, 6, 7, 7, 7]))

    Notes
    -----
    For an array ``a`` in ascending order, the resulting ``starts`` and ``stops``
    arrays have the same length as ``a`` and the property that, for all indices
    ``i``, ``[starts[i], stops[i])`` is the maximal range of indices ``j`` such
    that ``a[i] - radius <= a[j] <= a[i] + radius``.

    Index ranges are start-inclusive and stop-exclusive. This function is
    especially useful in conjunction with
    :meth:`.BlockMatrix.sparsify_row_intervals`.

    Parameters
    ----------
    a: :obj:`ndarray` of signed integer or float values
        1-dimensional array of values, non-decreasing with respect to index.
    radius: :obj:`float`
        Non-negative radius of window for values.

    Returns
    -------
    (:class:`ndarray` of :obj:`int64`, :class:`ndarray` of :obj:`int64`)
        Tuple of start indices array and stop indices array.
    """
    if radius < 0:
        raise ValueError(f'array_windows: radius must be non-negative, found {radius}')
    if a.ndim != 1:
        raise ValueError("array_windows: 'a' must be 1-dimensional")
    if not (np.issubdtype(a.dtype, np.signedinteger) or np.issubdtype(a.dtype, np.floating)):
        raise ValueError(f"array_windows: 'a' must be an ndarray of signed integer or float values, "
                         f"found dtype {str(a.dtype)}")

    size = a.size
    if size == 0:
        return np.zeros(shape=0, dtype=np.int64), np.zeros(shape=0, dtype=np.int64)

    if (not np.all(a[:-1] <= a[1:])) or np.isnan(a[0]):
        raise ValueError("array_windows: 'a' must be in ascending order with no nan elements")
    if a[0] - radius > a[0]:
        raise ValueError('array_windows: underflow for a[0] - radius')
    if a[-1] + radius < a[-1]:
        raise ValueError('array_windows: overflow for a[-1] + radius')

    starts, stops = np.zeros(size, dtype=np.int64),  np.zeros(size, dtype=np.int64)
    j, k = 0, 0
    for i in range(size):
        min_val = a[i] - radius
        while j < size and a[j] < min_val:
            j += 1
        starts[i] = j

        max_val = a[i] + radius
        while k < size and a[k] <= max_val:
            k += 1
        stops[i] = k

    return starts, stops


@typecheck(locus_expr=expr_locus(),
           radius=oneof(int, float),
           coord_expr=nullable(expr_float64))
def locus_windows(locus_expr, radius, coord_expr=None):
    """Returns start and stop indices for window around each locus.

    Examples
    --------

    Windows with 2bp radius for one contig with positions 1, 2, 3, 4, 5:

    >>> starts, stops = hl.locus_windows(
    ...     hl.balding_nichols_model(1, 5, 5).locus,
    ...     radius=2)
    >>> starts, stops
    (array([0, 0, 0, 1, 2]), array([3, 4, 5, 5, 5]))

    The following examples involve three contigs.

    >>> loci = [{'locus': hl.Locus('1', 1), 'cm': 1.0},
    ...         {'locus': hl.Locus('1', 2), 'cm': 3.0},
    ...         {'locus': hl.Locus('1', 4), 'cm': 4.0},
    ...         {'locus': hl.Locus('2', 1), 'cm': 2.0},
    ...         {'locus': hl.Locus('2', 1), 'cm': 2.0},
    ...         {'locus': hl.Locus('3', 3), 'cm': 5.0}]
    >>>
    >>> ht = hl.Table.parallelize(
    ...         loci,
    ...         hl.tstruct(locus=hl.tlocus('GRCh37'), cm=hl.tfloat64),
    ...         key=['locus'])

    Windows with 1bp radius:

    >>> hl.locus_windows(ht.locus, 1)
    (array([0, 0, 2, 3, 3, 5]), array([2, 2, 3, 5, 5, 6]))

    Windows with 1cm radius:

    >>> hl.locus_windows(ht.locus, 1.0, coord_expr=ht.cm)
    (array([0, 1, 1, 3, 3, 5]), array([1, 3, 3, 5, 5, 6]))

    Notes
    -----
    This function returns two 1-dimensional ndarrays of integers,
    ``starts`` and ``stops``, each of size equal to the number of rows.

    By default, for all indices ``i``, ``[starts[i], stops[i])`` is the maximal
    range of row indices ``j`` such that ``contig[i] == contig[j]`` and
    ``position[i] - radius <= position[j] <= position[i] + radius``.

    If the :meth:`.global_position` on `locus_expr` is not in ascending order,
    this method will fail. Ascending order should hold for a matrix table keyed
    by locus or variant (and the associated row table), or for a table that's
    been ordered by `locus_expr`.

    Set `coord_expr` to use a value other than position to define the windows.
    This row-indexed numeric expression must be non-missing, non-``nan``, on the
    same source as `locus_expr`, and ascending with respect to locus
    position for each contig; otherwise the function will fail.

    The last example above uses centimorgan coordinates, so
    ``[starts[i], stops[i])`` is the maximal range of row indices ``j`` such
    that ``contig[i] == contig[j]`` and
    ``cm[i] - radius <= cm[j] <= cm[i] + radius``.

    Index ranges are start-inclusive and stop-exclusive. This function is
    especially useful in conjunction with
    :meth:`.BlockMatrix.sparsify_row_intervals`.

    Parameters
    ----------
    locus_expr : :class:`.LocusExpression`
        Row-indexed locus expression on a table or matrix table.
    radius: :obj:`int`
        Radius of window for row values.
    coord_expr: :class:`.Float64Expression`, optional
        Row-indexed numeric expression for the row value.
        Must be on the same table or matrix table as `locus_expr`.
        By default, the row value is given by the locus position.

    Returns
    -------
    (:class:`ndarray` of :obj:`int64`, :class:`ndarray` of :obj:`int64`)
        Tuple of start indices array and stop indices array.
    """
    if radius < 0:
        raise ValueError(f"locus_windows: 'radius' must be non-negative, found {radius}")
    check_row_indexed('locus_windows', locus_expr)
    if coord_expr is None:
        global_pos_list = locus_expr.global_position().collect()
        n_loci = len(global_pos_list)
        global_pos = np.zeros(n_loci, dtype=np.int64)
        for i, p in enumerate(global_pos_list):
            if p is None:
                raise ValueError(f"locus_windows: missing value for 'locus_expr' global position at row {i}")
            global_pos[i] = p
        coord = global_pos
        del global_pos_list
    else:
        check_row_indexed('locus_windows', coord_expr)
        global_pos_and_coord =\
            hl.tuple([locus_expr.global_position(), coord_expr]).collect()  # raises exception if sources differ
        n_loci = len(global_pos_and_coord)

        global_pos = np.zeros(n_loci, dtype=np.int64)
        coord = np.zeros(n_loci, dtype=np.float64)
        for i, x in enumerate(global_pos_and_coord):
            if x[0] is None:
                raise ValueError(f"locus_windows: missing value for 'locus_expr' global position at row {i}")
            global_pos[i] = x[0]
            if x[1] is None:
                raise ValueError(f"locus_windows: missing value for 'coord_expr' at row {i}")
            coord[i] = x[1]
        del global_pos_and_coord

    if n_loci == 0:
        return np.zeros(shape=0, dtype=np.int64), np.zeros(shape=0, dtype=np.int64)

    contig_name = locus_expr.dtype.reference_genome.contigs
    contig_len = locus_expr.dtype.reference_genome.lengths
    contig_cum_len = np.cumsum([contig_len[name] for name in contig_name])

    assert(global_pos[-1] < contig_cum_len[-1])

    last = global_pos[0]
    contig_start_idx = [0]
    cum_len_iter = iter(contig_cum_len)
    cum_len = next(cum_len_iter)
    for i in range(n_loci):
        curr = global_pos[i]
        if curr < last:
            raise ValueError("locus_windows: 'locus_expr' global position must be in ascending order")
        while curr >= cum_len:
            contig_start_idx.append(i)
            cum_len = next(cum_len_iter)
        last = curr

    n_contigs = len(contig_start_idx)
    contig_start_idx.append(n_loci)
    contig_bounds = [array_windows(coord[contig_start_idx[c]:contig_start_idx[c + 1]], radius)
                     for c in range(n_contigs)]
    starts = np.concatenate([contig_start_idx[c] + contig_bounds[c][0] for c in range(n_contigs)])
    stops = np.concatenate([contig_start_idx[c] + contig_bounds[c][1] for c in range(n_contigs)])

    return starts, stops
