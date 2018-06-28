
import numpy as np
import hail as hl
from hail.table import Table
from hail.linalg import BlockMatrix
from hail.typecheck import *
from hail.expr.expressions import *
from hail.utils import new_temp_file, wrap_to_list

@typecheck(entry_expr=expr_numeric,
           annotation_exprs=oneof(expr_numeric, sequenceof(expr_numeric)),
           position_expr=expr_numeric,
           window_size=oneof(int, float))
def ld_score(entry_expr, annotation_exprs, position_expr, window_size) -> Table:
    """
    Calculate LD scores.

    Example
    -------
    
    >>> # Load genetic data into MatrixTable
    >>> mt = hl.import_plink(bed='gs://.../my_data.bed', 
    ...                      bim='gs://.../my_data.bim', 
    ...                      fam='gs://.../my_data.fam')
    
    >>> # Create locus-keyed Table with numeric variant annotations
    >>> ht = hl.import_table('gs://.../my_annotations.tsv.bgz')
    >>> ht = ht.annotate(locus=hl.locus(ht.contig, ht.position))  
    >>> ht = ht.key_by('locus')
    
    >>> # Annotate MatrixTable with external annotations 
    >>> mt = mt.annotate_rows(annotation_0=hl.literal(1),   # univariate LD scores
    ...                       annotation_1=ht_annotations[mt.locus].annotation_1, 
    ...                       annotation_2=ht_annotations[mt.locus].annotation_2)
    ...
    >>> # Annotate MatrixTable with alt allele count stats
    >>> mt = mt.annotate_rows(stats=hl.agg.stats(mt.GT.n_alt_alleles()))
     
    >>> # Calculate LD score for each variant, annotation using standardized genotypes
    >>> ht_scores = hl.ld_score(entry_expr=hl.or_else((mt.GT.n_alt_alleles() - mt.stats.mean)/mt.stats.stdev, 0.0),
    ...                         annotation_exprs=[mt.annotation_0, mt.annotation_1, mt.annotation_2],
    ...                         position_expr=mt.cm_position,
    ...                         window_size=1)

    Warning
    -------
    :func:`.ld_score` will fail if ``entry_expr`` results in any missing values. The special float value ``nan`` is not considered a missing value.


    **Further reading**

    For more in-depth discussion of LD scores, see:

    - `LD Score regression distinguishes confounding from polygenicity in genome-wide association studies (Bulik-Sullivan et al, 2015) <https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4495769/>`__
    - `Partitioning heritability by functional annotation using genome-wide association summary statistics (Finucane et al, 2015) <https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4626285/>`__

    Parameters
    ----------
    entry_expr  : :class:`.NumericExpression`
        Expression for entries of genotype matrix (e.g. ``mt.GT.n_alt_alleles()``).
    annotation_exprs : :class:`.NumericExpression` or :obj:`list` of :class:`.NumericExpression`
        Annotation expression(s) to partition LD scores.
    position_expr   : :class:`.NumericExpression`
        Expression for position of variant (e.g. ``mt.cm_position``, ``mt.locus.position``).
    window_size : :obj:`int` or :obj:`float`
        Size of variant window used to calculate LD scores, in units of ``position``.

    Returns
    -------
    :class:`.HailTable` 
        Locus-keyed table with LD scores for each variant and annotation.
    """

    assert window_size >= 0

    mt = entry_expr._indices.source
    annotations = wrap_to_list(annotation_exprs)
    variant_key = [x for x in mt.row_key]

    ht_annotations = mt.select_rows(*annotations).rows()
    annotation_names = [x for x in ht_annotations.row if x not in variant_key]

    ht_annotations = hl.Table.union(*[(ht_annotations.annotate(annotation=hl.str(x), value=hl.float(ht_annotations[x]))
                                                     .select('annotation', 'value')) for x in annotation_names])
    mt_annotations = ht_annotations.to_matrix_table(row_key=variant_key, col_key=['annotation'])

    cols = mt_annotations['annotation'].collect()
    col_idxs = {i: cols[i] for i in range(len(cols))}

    G = BlockMatrix.from_entry_expr(entry_expr)
    A = BlockMatrix.from_entry_expr(mt_annotations.value)

    n = G.n_cols

    R_squared = ((G @ G.T)/n) ** 2
    R_squared_adj = R_squared - (1.0 - R_squared)/(n - 2.0)
    
    positions = [(x[0], float(x[1])) for x in hl.array([mt.locus.contig, hl.str(position_expr)]).collect()]
    n_positions = len(positions)

    starts = np.zeros(n_positions, dtype=int)
    stops = np.zeros(n_positions, dtype=int)

    contig = '0'
    
    for i, (c, p) in enumerate(positions):

        if c != contig:

            j = i
            k = i
            contig = c

        min_val = p - window_size
        max_val = p + window_size

        while j < n_positions and positions[j][1] < min_val:
            j += 1

        starts[i] = j

        if k == n_positions:
            stops[i] = k
            continue

        while positions[k][0] == contig and positions[k][1] <= max_val:
            k += 1
            if k == n_positions:
                break

        stops[i] = k

    R_squared_adj_sparse = R_squared_adj.sparsify_row_intervals(starts=[int(x) for x in starts], stops=[int(x) for x in stops])
    L_squared = R_squared_adj_sparse @ A

    tmp_bm_path = new_temp_file()
    tmp_tsv_path = new_temp_file()

    L_squared.write(tmp_bm_path, force_row_major=True)
    BlockMatrix.export(tmp_bm_path, tmp_tsv_path)

    ht_scores = hl.import_table(tmp_tsv_path, no_header=True, impute=True)
    ht_scores = ht_scores.add_index()
    ht_scores = ht_scores.key_by('idx')
    ht_scores = ht_scores.rename({'f{:}'.format(i): col_idxs[i] for i in range(len(cols))})

    ht_variants = mt.rows()
    ht_variants = ht_variants.drop(*[x for x in ht_variants.row if x not in variant_key])
    ht_variants = ht_variants.add_index()
    ht_variants = ht_variants.key_by('idx')

    ht_scores = ht_variants.join(ht_scores, how='inner')
    ht_scores = ht_scores.key_by('locus')
    ht_scores = ht_scores.drop('alleles', 'idx')

    return ht_scores
