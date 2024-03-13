from typing import Union

import hail as hl
from hail import ir
from hail.expr import Expression, construct_expr, construct_variable, expr_any, expr_array, expr_interval, expr_numeric
from hail.expr.types import tarray, tlocus, tstr, tstruct, ttuple
from hail.matrixtable import MatrixTable
from hail.table import Table
from hail.typecheck import func_spec, nullable, oneof, typecheck
from hail.utils import Interval, Struct, deduplicate, new_temp_file
from hail.utils.java import Env, info
from hail.utils.misc import plural


@typecheck(i=Expression, j=Expression, keep=bool, tie_breaker=nullable(func_spec(2, expr_numeric)), keyed=bool)
def maximal_independent_set(i, j, keep=True, tie_breaker=None, keyed=True) -> Table:
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
    ...     return hl.if_else(l.is_case & ~r.is_case, -1,
    ...                       hl.if_else(~l.is_case & r.is_case, 1, 0))
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
    `r`---each of which is an :class:`~.Expression` of the same type as `i` and
    `j`. `tie_breaker` returns a :class:`~.NumericExpression`, which defines an
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

    If `keyed` is ``False``, then a node may appear twice in the resulting
    table.

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
    keyed : :obj:`bool`
        If ``True``, key the resulting table by the `node` field, this requires
        a sort.

    Returns
    -------
    :class:`.Table`
        Table with the set of independent vertices. The table schema is one row
        field `node` which has the same type as input expressions `i` and `j`.
    """

    if i.dtype != j.dtype:
        raise ValueError(
            "'maximal_independent_set' expects arguments `i` and `j` to have same type. " "Found {} and {}.".format(
                i.dtype, j.dtype
            )
        )

    source = i._indices.source
    if not isinstance(source, Table):
        raise ValueError(
            "'maximal_independent_set' expects an expression of 'Table'. Found {}".format(
                "expression of '{}'".format(source.__class__) if source is not None else 'scalar expression'
            )
        )

    if i._indices.source != j._indices.source:
        raise ValueError(
            "'maximal_independent_set' expects arguments `i` and `j` to be expressions of the same Table. "
            "Found\n{}\n{}".format(i, j)
        )

    node_t = i.dtype

    if tie_breaker:
        wrapped_node_t = ttuple(node_t)
        left_id = Env.get_uid()
        right_id = Env.get_uid()
        left = construct_variable(left_id, wrapped_node_t)
        right = construct_variable(right_id, wrapped_node_t)
        tie_breaker_expr = hl.float64(tie_breaker(left[0], right[0]))
        tie_breaker_ir = tie_breaker_expr._ir
        t, _ = source._process_joins(i, j, tie_breaker_expr)
    else:
        left_id, right_id, tie_breaker_ir = None, None, None
        t, _ = source._process_joins(i, j)

    edges = t.select(__i=i, __j=j).key_by().select('__i', '__j')
    edges = edges.checkpoint(new_temp_file())

    mis_nodes = hl.set(
        construct_expr(
            ir.ArrayMaximalIndependentSet(edges.collect(_localize=False)._ir, left_id, right_id, tie_breaker_ir),
            hl.tarray(node_t),
        )
    )

    nodes = edges.select(node=[edges.__i, edges.__j])
    nodes = nodes.explode(nodes.node)
    nodes = nodes.annotate_globals(mis_nodes=mis_nodes)
    nodes = nodes.filter(nodes.mis_nodes.contains(nodes.node), keep)
    nodes = nodes.select_globals()
    if keyed:
        return nodes.key_by('node').distinct()
    return nodes


def require_col_key_str(dataset: MatrixTable, method: str):
    if not len(dataset.col_key) == 1 or dataset[next(iter(dataset.col_key))].dtype != hl.tstr:
        raise ValueError(
            f"Method '{method}' requires column key to be one field of type 'str', found "
            f"{list(str(x.dtype) for x in dataset.col_key.values())}"
        )


def require_table_key_variant(ht, method):
    if (
        list(ht.key) != ['locus', 'alleles']
        or not isinstance(ht['locus'].dtype, tlocus)
        or not ht['alleles'].dtype == tarray(tstr)
    ):
        raise ValueError(
            "Method '{}' requires key to be two fields 'locus' (type 'locus<any>') and "
            "'alleles' (type 'array<str>')\n"
            "  Found:{}".format(method, ''.join("\n    '{}': {}".format(k, str(ht[k].dtype)) for k in ht.key))
        )


def require_row_key_variant(dataset, method):
    if isinstance(dataset, Table):
        key = dataset.key
    else:
        assert isinstance(dataset, MatrixTable)
        key = dataset.row_key
    if (
        list(key) != ['locus', 'alleles']
        or not isinstance(dataset['locus'].dtype, tlocus)
        or not dataset['alleles'].dtype == tarray(tstr)
    ):
        raise ValueError(
            "Method '{}' requires row key to be two fields 'locus' (type 'locus<any>') and "
            "'alleles' (type 'array<str>')\n"
            "  Found:{}".format(method, ''.join("\n    '{}': {}".format(k, str(dataset[k].dtype)) for k in key))
        )


def require_alleles_field(dataset, method):
    if 'alleles' not in dataset.row:
        raise ValueError(f"Method '{method}' requires a field 'alleles' (type 'array<str>')\n")
    if dataset.alleles.dtype != tarray(tstr):
        raise ValueError(
            f"Method '{method}' requires a field 'alleles' (type 'array<str>')\n"
            f"  Found:\n"
            f"    'alleles': {dataset.alleles.dtype}"
        )


def require_row_key_variant_w_struct_locus(dataset, method):
    if (
        list(dataset.row_key) != ['locus', 'alleles']
        or not dataset['alleles'].dtype == tarray(tstr)
        or (
            not isinstance(dataset['locus'].dtype, tlocus)
            and dataset['locus'].dtype != hl.dtype('struct{contig: str, position: int32}')
        )
    ):
        raise ValueError(
            "Method '{}' requires row key to be two fields 'locus'"
            " (type 'locus<any>' or 'struct{{contig: str, position: int32}}') and "
            "'alleles' (type 'array<str>')\n"
            "  Found:{}".format(
                method, ''.join("\n    '{}': {}".format(k, str(dataset[k].dtype)) for k in dataset.row_key)
            )
        )


def require_first_key_field_locus(dataset, method):
    if isinstance(dataset, Table):
        key = dataset.key
    else:
        assert isinstance(dataset, MatrixTable)
        key = dataset.row_key
    if len(key) == 0 or not isinstance(key[0].dtype, tlocus):
        raise ValueError(
            "Method '{}' requires first key field of type 'locus<any>'.\n" "  Found:{}".format(
                method, ''.join("\n    '{}': {}".format(k, str(dataset[k].dtype)) for k in key)
            )
        )


@typecheck(table=Table, method=str)
def require_key(table, method):
    if len(table.key) == 0:
        raise ValueError("Method '{}' requires a non-empty key".format(method))


@typecheck(dataset=MatrixTable, method=str, tolerate_generic_locus=bool)
def require_biallelic(dataset, method, tolerate_generic_locus: bool = False) -> MatrixTable:
    if tolerate_generic_locus:
        require_row_key_variant_w_struct_locus(dataset, method)
    else:
        require_row_key_variant(dataset, method)
    return dataset._select_rows(
        method,
        hl.case()
        .when(dataset.alleles.length() == 2, dataset._rvrow)
        .or_error(
            f"'{method}' expects biallelic variants ('alleles' field of length 2), found "
            + hl.str(dataset.locus)
            + ", "
            + hl.str(dataset.alleles)
        ),
    )


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
    name : :class:`str`
        Name of new field.

    Returns
    -------
    :class:`.MatrixTable`
    """

    require_col_key_str(dataset, 'rename_duplicates')
    ids = dataset.col_key[0].collect()

    mapping, new_ids = deduplicate(ids)

    if mapping:
        info(
            f'Renamed {len(mapping)} duplicate {plural("sample ID", len(mapping))}. Mangled IDs as follows:'
            + ''.join(f'\n  "{pre}" => "{post}"' for pre, post in mapping)
        )
    else:
        info('No duplicate sample IDs found.')
    return dataset.annotate_cols(**{name: hl.literal(new_ids)[hl.int(hl.scan.count())]})


@typecheck(ds=oneof(Table, MatrixTable), intervals=expr_array(expr_interval(expr_any)), keep=bool)
def filter_intervals(ds, intervals, keep=True) -> Union[Table, MatrixTable]:
    """Filter rows with a list of intervals.

    Examples
    --------

    Filter to loci falling within one interval:

    >>> ds_result = hl.filter_intervals(dataset, [hl.parse_locus_interval('17:38449840-38530994')])

    Remove all loci within list of intervals:

    >>> intervals = [hl.parse_locus_interval(x) for x in ['1:50M-75M', '2:START-400000', '3-22']]
    >>> ds_result = hl.filter_intervals(dataset, intervals, keep=False)

    Notes
    -----
    Based on the `keep` argument, this method will either restrict to points
    in the supplied interval ranges, or remove all rows in those ranges.

    When ``keep=True``, partitions that don't overlap any supplied interval
    will not be loaded at all.  This enables :func:`.filter_intervals` to be
    used for reasonably low-latency queries of small ranges of the dataset, even
    on large datasets.

    Parameters
    ----------
    ds : :class:`.MatrixTable` or :class:`.Table`
        Dataset to filter.
    intervals : :class:`.ArrayExpression` of type :class:`.tinterval`
        Intervals to filter on.  The point type of the interval must
        be a prefix of the key or equal to the first field of the key.
    keep : :obj:`bool`
        If ``True``, keep only rows that fall within any interval in `intervals`.
        If ``False``, keep only rows that fall outside all intervals in
        `intervals`.

    Returns
    -------
    :class:`.MatrixTable` or :class:`.Table`

    """

    if isinstance(ds, MatrixTable):
        k_type = ds.row_key.dtype
    else:
        assert isinstance(ds, Table)
        k_type = ds.key.dtype

    point_type = intervals.dtype.element_type.point_type

    def is_struct_prefix(partial, full):
        if list(partial) != list(full)[: len(partial)]:
            return False
        for k, v in partial.items():
            if full[k] != v:
                return False
        return True

    if point_type == k_type[0]:
        needs_wrapper = True
        k_name = k_type.fields[0]
        point_type = hl.tstruct(**{k_name: k_type[k_name]})
    elif isinstance(point_type, tstruct) and is_struct_prefix(point_type, k_type):
        needs_wrapper = False
    else:
        raise TypeError(
            "The point type is incompatible with key type of the dataset ('{}', '{}')".format(
                repr(point_type), repr(k_type)
            )
        )

    def wrap_input(interval):
        if interval is None:
            raise TypeError("'filter_intervals' does not allow missing values in 'intervals'.")
        elif needs_wrapper:
            return Interval(
                Struct(**{k_name: interval.start}),
                Struct(**{k_name: interval.end}),
                interval.includes_start,
                interval.includes_end,
            )
        else:
            return interval

    intervals = hl.eval(intervals)
    intervals = [wrap_input(i) for i in intervals]

    if isinstance(ds, MatrixTable):
        return MatrixTable(ir.MatrixFilterIntervals(ds._mir, intervals, point_type, keep))
    else:
        return Table(ir.TableFilterIntervals(ds._tir, intervals, point_type, keep))


@typecheck(ht=Table, points=oneof(Table, expr_array(expr_any)))
def segment_intervals(ht, points):
    """Segment the interval keys of `ht` at a given set of points.

    Parameters
    ----------
    ht : :class:`.Table`
        Table with interval keys.
    points : :class:`.Table` or :class:`.ArrayExpression`
        Points at which to segment the intervals, a table or an array.

    Returns
    -------
    :class:`.Table`
    """
    if len(ht.key) != 1 or not isinstance(ht.key[0].dtype, hl.tinterval):
        raise ValueError("'segment_intervals' expects a table with interval keys")
    point_type = ht.key[0].dtype.point_type
    if isinstance(points, Table):
        if len(points.key) != 1 or points.key[0].dtype != point_type:
            raise ValueError(
                "'segment_intervals' expects points to be a table with a single"
                " key of the same type as the intervals in 'ht', or an array of those points:"
                f"\n  expect {point_type}, found {list(points.key.dtype.values())}"
            )
        points = hl.array(hl.set(points.collect(_localize=False)))
    if points.dtype.element_type != point_type:
        raise ValueError(
            f"'segment_intervals' expects points to be a table with a single"
            f" key of the same type as the intervals in 'ht', or an array of those points:"
            f"\n  expect {point_type}, found {points.dtype.element_type}"
        )

    points = hl._sort_by(points, lambda l, r: hl._compare(l, r) < 0)

    ht = ht.annotate_globals(__points=points)

    interval = ht.key[0]
    points = ht.__points
    lower = hl.expr.functions._lower_bound(points, interval.start)
    higher = hl.expr.functions._lower_bound(points, interval.end)
    n_points = hl.len(points)
    lower = hl.if_else((lower < n_points) & (points[lower] == interval.start), lower + 1, lower)
    higher = hl.if_else((higher < n_points) & (points[higher] == interval.end), higher - 1, higher)
    interval_results = hl.rbind(
        lower,
        higher,
        lambda lower, higher: hl.if_else(
            lower >= higher,
            [interval],
            hl.flatten([
                [
                    hl.interval(
                        interval.start, points[lower], includes_start=interval.includes_start, includes_end=False
                    )
                ],
                hl.range(lower, higher - 1).map(
                    lambda x: hl.interval(points[x], points[x + 1], includes_start=True, includes_end=False)
                ),
                [
                    hl.interval(
                        points[higher - 1], interval.end, includes_start=True, includes_end=interval.includes_end
                    )
                ],
            ]),
        ),
    )
    ht = ht.annotate(__new_intervals=interval_results, lower=lower, higher=higher).explode('__new_intervals')
    return ht.key_by(**{list(ht.key)[0]: ht.__new_intervals}).drop('__new_intervals')
