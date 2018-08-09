
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

    Warning
    -------
        :func:`.ld_score` will fail if ``entry_expr`` results in any missing
        values. The special float value ``nan`` is not considered a
        missing value.

    **Further reading**

    For more in-depth discussion of LD scores, see:

    - `LD Score regression distinguishes confounding from polygenicity in genome-wide association studies (Bulik-Sullivan et al, 2015) <https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4495769/>`__
    - `Partitioning heritability by functional annotation using genome-wide association summary statistics (Finucane et al, 2015) <https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4626285/>`__

    Parameters
    ----------
    entry_expr : :class:`.NumericExpression`
        Expression for entries of genotype matrix
        (e.g. ``mt.GT.n_alt_alleles()``).
    locus_expr : :class:`.LocusExpression`
        Row-indexed locus expression on a table or matrix table that is
        row-aligned with the matrix table of `entry_expr`.
    radius : :obj:`int` or :obj:`float`
        Radius of window for row values (in units of `coord_expr`).
    annotation_exprs : :class:`.NumericExpression` or
                       :obj:`list` of :class:`.NumericExpression`, optional
        Annotation expression(s) to partition LD scores. Univariate
        annotation will always be included and does not need to be
        specified.
    coord_expr: :class:`.Float64Expression`, optional
        Row-indexed numeric expression for the row value on the same table or
        matrix table as `locus_expr`.
        By default, the row value is given by the locus position.
    block_size : :obj:`int`, optional
        Block size. Default given by :meth:`.BlockMatrix.default_block_size`.

    Returns
    -------
    :class:`.Table`
        Table keyed by `locus_expr` with LD scores for each variant and
        `annotation_expr`. The function will always return LD scores for
        the univariate (all SNPs) annotation."""

    mt = entry_expr._indices.source
    N = mt.count_cols()

    R2 = hl.row_correlation(entry_expr, block_size) ** 2
    R2_adj = (N - 1.0)/(N - 2.0) * R2 - 1.0/(N - 2.0)

    starts, stops = hl.linalg.utils.locus_windows(locus_expr,
                                                  radius,
                                                  coord_expr)
    R2_adj_sparse = R2_adj.sparsify_row_intervals(starts, stops)

    tmp_R2_adj_sparse_file = new_temp_file()
    R2_adj_sparse.write(tmp_R2_adj_sparse_file, force_row_major=True)
    R2_adj_sparse = BlockMatrix.read(tmp_R2_adj_sparse_file)

    if not annotation_exprs:
        cols = ['univariate']
        col_idxs = {0: 'univariate'}
        L2 = R2_adj_sparse.sum(axis=1)
    else:
        ht = mt.select_rows(*wrap_to_list(annotation_exprs)).rows()
        ht = ht.annotate(univariate=hl.literal(1.0))
        names = [x for x in ht.row if x not in ht.key]

        ht_union = hl.Table.union(
            *[(ht.annotate(name=hl.str(x),
                           value=hl.float(ht[x]))
                 .select('name', 'value')) for x in names])
        mt_annotations = ht_union.to_matrix_table(
            row_key=[x for x in ht_union.key],
            col_key=['name'])

        cols = mt_annotations['name'].collect()
        col_idxs = {i: cols[i] for i in range(len(cols))}

        A = BlockMatrix.from_entry_expr(mt_annotations.value)
        tmp_A_file = new_temp_file()
        BlockMatrix.write_from_entry_expr(mt_annotations.value, tmp_A_file)

        A = BlockMatrix.read(tmp_A_file)
        L2 = R2_adj_sparse @ A

    tmp_L2_bm_file = new_temp_file()
    tmp_L2_tsv_file = new_temp_file()
    L2.write(tmp_L2_bm_file, force_row_major=True)
    BlockMatrix.export(tmp_L2_bm_file, tmp_L2_tsv_file)

    ht_scores = hl.import_table(tmp_L2_tsv_file, no_header=True, impute=True)
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
