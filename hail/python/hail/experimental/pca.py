import hail as hl
from hail.typecheck import typecheck
from hail.expr.expressions import (
    expr_call,
    expr_numeric,
    expr_array,
    raise_unless_entry_indexed,
    raise_unless_row_indexed,
)


@typecheck(call_expr=expr_call, loadings_expr=expr_array(expr_numeric), af_expr=expr_numeric)
def pc_project(call_expr, loadings_expr, af_expr):
    """Projects genotypes onto pre-computed PCs. Requires loadings and
    allele-frequency from a reference dataset (see example). Note that
    `loadings_expr` must have no missing data and reflect the rows
    from the original PCA run for this method to be accurate.

    Example
    -------
    >>> # Compute loadings and allele frequency for reference dataset
    >>> _, _, loadings_ht = hl.hwe_normalized_pca(mt.GT, k=10, compute_loadings=True)   # doctest: +SKIP
    >>> mt = mt.annotate_rows(af=hl.agg.mean(mt.GT.n_alt_alleles()) / 2)                # doctest: +SKIP
    >>> loadings_ht = loadings_ht.annotate(af=mt.rows()[loadings_ht.key].af)            # doctest: +SKIP
    >>> # Project new genotypes onto loadings
    >>> ht = pc_project(mt_to_project.GT, loadings_ht.loadings, loadings_ht.af)         # doctest: +SKIP

    Parameters
    ----------
    call_expr : :class:`.CallExpression`
        Entry-indexed call expression for genotypes
        to project onto loadings.
    loadings_expr : :class:`.ArrayNumericExpression`
        Location of expression for loadings
    af_expr : :class:`.Float64Expression`
        Location of expression for allele frequency

    Returns
    -------
    :class:`.Table`
        Table with scores calculated from loadings in column `scores`
    """
    raise_unless_entry_indexed('pc_project', call_expr)
    raise_unless_row_indexed('pc_project', loadings_expr)
    raise_unless_row_indexed('pc_project', af_expr)

    gt_source = call_expr._indices.source
    loadings_source = loadings_expr._indices.source
    af_source = af_expr._indices.source

    loadings_expr = _get_expr_or_join(loadings_expr, loadings_source, gt_source, '_loadings')
    af_expr = _get_expr_or_join(af_expr, af_source, gt_source, '_af')

    mt = gt_source._annotate_all(
        row_exprs={'_loadings': loadings_expr, '_af': af_expr}, entry_exprs={'_call': call_expr}
    )

    if isinstance(loadings_source, hl.MatrixTable):
        n_variants = loadings_source.count_rows()
    else:
        n_variants = loadings_source.count()

    mt = mt.filter_rows(hl.is_defined(mt._loadings) & hl.is_defined(mt._af) & (mt._af > 0) & (mt._af < 1))

    gt_norm = (mt._call.n_alt_alleles() - 2 * mt._af) / hl.sqrt(n_variants * 2 * mt._af * (1 - mt._af))

    return mt.select_cols(scores=hl.agg.array_sum(mt._loadings * gt_norm)).cols()


def _get_expr_or_join(expr, source, other_source, loc):
    if source != other_source:
        if isinstance(source, hl.MatrixTable):
            source = source.annotate_rows(**{loc: expr})
        else:
            source = source.annotate(**{loc: expr})
        expr = source[other_source.row_key][loc]
    return expr
