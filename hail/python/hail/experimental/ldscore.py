
import numpy as np
import hail as hl
from hail.table import Table
from hail.linalg import BlockMatrix
from hail.typecheck import *
from hail.expr.expressions import *
from hail.utils import new_temp_file, wrap_to_list


@typecheck(entry_expr=expr_float64,
           locus_expr=expr_locus(),
           radius=oneof(int, float),
           coord_expr=nullable(expr_float64),
           annotation_exprs=nullable(oneof(expr_numeric,
                                           sequenceof(expr_numeric))),
           block_size=nullable(int))
def ld_score(entry_expr,
             locus_expr,
             radius,
             coord_expr=None,
             annotation_exprs=None,
             block_size=None) -> Table:
    """Calculate LD scores.

    Example
    -------

    >>> # Load genetic data into MatrixTable
    >>> mt = hl.import_plink(bed='data/ldsc.bed',
    ...                      bim='data/ldsc.bim',
    ...                      fam='data/ldsc.fam')

    >>> # Create locus-keyed Table with numeric variant annotations
    >>> ht = hl.import_table('data/ldsc.annot',
    ...                      types={'BP': hl.tint,
    ...                             'binary': hl.tfloat,
    ...                             'continuous': hl.tfloat})
    >>> ht = ht.annotate(locus=hl.locus(ht.CHR, ht.BP))
    >>> ht = ht.key_by('locus')

    >>> # Annotate MatrixTable with external annotations
    >>> mt = mt.annotate_rows(binary_annotation=ht[mt.locus].binary,
    ...                       continuous_annotation=ht[mt.locus].continuous)

    >>> # Calculate LD scores using centimorgan coordinates
    >>> ht_scores = hl.experimental.ld_score(entry_expr=mt.GT.n_alt_alleles(),
    ...                                      locus_expr=mt.locus,
    ...                                      radius=1.0,
    ...                                      coord_expr=mt.cm_position,
    ...                                      annotation_exprs=[mt.binary_annotation,
    ...                                                        mt.continuous_annotation])

    >>> # Show results
    >>> ht_scores.show(3)

    .. code-block:: text

        +---------------+-------------------+-----------------------+-------------+
        | locus         | binary_annotation | continuous_annotation |  univariate |
        +---------------+-------------------+-----------------------+-------------+
        | locus<GRCh37> |           float64 |               float64 |     float64 |
        +---------------+-------------------+-----------------------+-------------+
        | 20:82079      |       1.15183e+00 |           7.30145e+01 | 1.60117e+00 |
        | 20:103517     |       2.04604e+00 |           2.75392e+02 | 4.69239e+00 |
        | 20:108286     |       2.06585e+00 |           2.86453e+02 | 5.00124e+00 |
        +---------------+-------------------+-----------------------+-------------+


    Warning
    -------
        :func:`.ld_score` will fail if ``entry_expr`` results in any missing
        values. The special float value ``nan`` is not considered a
        missing value.

    **Further reading**

    For more in-depth discussion of LD scores, see:

    - `LD Score regression distinguishes confounding from polygenicity in genome-wide association studies (Bulik-Sullivan et al, 2015) <https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4495769/>`__
    - `Partitioning heritability by functional annotation using genome-wide association summary statistics (Finucane et al, 2015) <https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4626285/>`__

    Notes
    -----

    `entry_expr`, `locus_expr`, `coord_expr` (if specified), and
    `annotation_exprs` (if specified) must come from the same
    MatrixTable.


    Parameters
    ----------
    entry_expr : :class:`.NumericExpression`
        Expression for entries of genotype matrix
        (e.g. ``mt.GT.n_alt_alleles()``).
    locus_expr : :class:`.LocusExpression`
        Row-indexed locus expression.
    radius : :obj:`int` or :obj:`float`
        Radius of window for row values (in units of `coord_expr` if set,
        otherwise in units of basepairs).
    coord_expr: :class:`.Float64Expression`, optional
        Row-indexed numeric expression for the row value used to window
        variants. By default, the row value is given by the locus
        position.
    annotation_exprs : :class:`.NumericExpression` or
                       :obj:`list` of :class:`.NumericExpression`, optional
        Annotation expression(s) to partition LD scores. Univariate
        annotation will always be included and does not need to be
        specified.
    block_size : :obj:`int`, optional
        Block size. Default given by :meth:`.BlockMatrix.default_block_size`.

    Returns
    -------
    :class:`.Table`
        Table keyed by `locus_expr` with LD scores for each variant and
        `annotation_expr`. The function will always return LD scores for
        the univariate (all SNPs) annotation."""

    mt = entry_expr._indices.source
    mt_locus_expr = locus_expr._indices.source

    if coord_expr is None:
        mt_coord_expr = mt_locus_expr
    else:
        mt_coord_expr = coord_expr._indices.source

    if not annotation_exprs:
        check_mts = all([mt == mt_locus_expr,
                         mt == mt_coord_expr])
    else:
        check_mts = all([mt == mt_locus_expr,
                         mt == mt_coord_expr] +
                        [mt == x._indices.source
                         for x in wrap_to_list(annotation_exprs)])

    if not check_mts:
        raise ValueError("""ld_score: entry_expr, locus_expr, coord_expr
                            (if specified), and annotation_exprs (if
                            specified) must come from same MatrixTable.""")

    n = mt.count_cols()
    r2 = hl.row_correlation(entry_expr, block_size) ** 2
    r2_adj = ((n-1.0) / (n-2.0)) * r2 - (1.0 / (n-2.0))

    starts, stops = hl.linalg.utils.locus_windows(locus_expr,
                                                  radius,
                                                  coord_expr)
    r2_adj_sparse = r2_adj.sparsify_row_intervals(starts, stops)

    r2_adj_sparse_tmp = new_temp_file()
    r2_adj_sparse.write(r2_adj_sparse_tmp)
    r2_adj_sparse = BlockMatrix.read(r2_adj_sparse_tmp)

    if not annotation_exprs:
        cols = ['univariate']
        col_idxs = {0: 'univariate'}
        l2 = r2_adj_sparse.sum(axis=1)
    else:
        ht = mt.select_rows(*wrap_to_list(annotation_exprs)).rows()
        ht = ht.annotate(univariate=hl.literal(1.0))
        names = [name for name in ht.row if name not in ht.key]

        ht_union = hl.Table.union(
            *[(ht.annotate(name=hl.str(x),
                           value=hl.float(ht[x]))
                 .select('name', 'value')) for x in names])
        mt_annotations = ht_union.to_matrix_table(
            row_key=list(ht_union.key),
            col_key=['name'])

        cols = mt_annotations.key_cols_by()['name'].collect()
        col_idxs = {i: cols[i] for i in range(len(cols))}

        a_tmp = new_temp_file()
        BlockMatrix.write_from_entry_expr(mt_annotations.value, a_tmp)

        a = BlockMatrix.read(a_tmp)
        l2 = r2_adj_sparse @ a

    l2_bm_tmp = new_temp_file()
    l2_tsv_tmp = new_temp_file()
    l2.write(l2_bm_tmp, force_row_major=True)
    BlockMatrix.export(l2_bm_tmp, l2_tsv_tmp)

    ht_scores = hl.import_table(l2_tsv_tmp, no_header=True, impute=True)
    ht_scores = ht_scores.add_index()
    ht_scores = ht_scores.key_by('idx')
    ht_scores = ht_scores.rename({'f{:}'.format(i): col_idxs[i]
                                  for i in range(len(cols))})

    ht = mt.select_rows(__locus=locus_expr).rows()
    ht = ht.add_index()
    ht = ht.annotate(**ht_scores[ht.idx])
    ht = ht.key_by('__locus')
    ht = ht.select(*[x for x in ht_scores.row if x not in ht_scores.key])
    ht = ht.rename({'__locus': 'locus'})

    return ht
