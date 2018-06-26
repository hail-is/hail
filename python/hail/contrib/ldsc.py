import numpy as np
import hail as hl
from hail.linalg import BlockMatrix
from hail.typecheck import *
from hail.expr.expressions import *
from hail.utils import new_temp_file

def _compute_row_intervals(positions, window_size):
    """
    a is ndarray of non-decreasing floats
    b is ndarray of non-decreasing floats
    radius is non-negative float
    
    starts and stops are ndarrays of int of length a.size
    starts[i] is minimum index j in b such that b[j] >= a[i] - radius
    stops[i] is one greater than the maximum index j in b such that b[j] <= a[i] + radius
    """ 
    assert window_size >= 0
    
    positions_size = positions.size

    starts = np.zeros(positions_size, dtype=int)
    stops = np.zeros(positions_size, dtype=int)
    
    j = 0
    k = 0
    
    for i in range(positions_size):
        min_val = positions[i] - window_size
        max_val = positions[i] + window_size
        while (j < positions_size) and (positions[j] < min_val):
            j += 1
        starts[i] = j
        while (k < positions_size) and (positions[k] <= max_val):
            k += 1
        stops[i] = k

    return [int(x) for x in starts], [int(x) for x in stops]

@typecheck(entry_expr=expr_numeric,
           annotations=oneof(expr_numeric, sequenceof(expr_numeric)),
           position=expr_numeric,
           window_size=oneof(int, float))
def ld_score(entry_expr, annotations, position, window_size):
    """
    Calculate LD scores.

    Example
    -------
    # Load genetic data into MatrixTable
    mt = hl.import_plink(bed='gs://.../my_data.bed',
                         bim='gs://.../my_data.bim',
                         fam='gs://.../my_data.fam')

    # Create locus-keyed Table with numeric variant annotations
    ht = hl.import_table('gs://.../my_annotations.tsv.bgz')
    ht = ht.annotate(locus=hl.locus(ht.contig, ht.position))  
    ht = ht.key_by('locus')

    # Annotate MatrixTable with external annotations 
    mt = mt.annotate_rows(annotation_0=hl.literal(1),   # univariate LD scores
                          annotation_1=ht_annotations[mt.locus].annotation_1, 
                          annotation_2=ht_annotations[mt.locus].annotation_2)

    # Annotate MatrixTable with alt allele count stats
    mt = mt.annotate_rows(stats=hl.agg.stats(mt.GT.n_alt_alleles()))

    # Calculate LD score for each variant, annotation using standardized genotypes
    ht_scores = hl.ld_score(entry_expr=hl.or_else((mt.GT.n_alt_alleles() - mt.stats.mean)/mt.stats.stdev, 0.0),
                            annotations=[mt.annotation_0, mt.annotation_1, mt.annotation_2],
                            position=mt.cm_position,
                            window_size=1)

    Warning
    -------
    :func:`.ld_score` will fail if `entry_expr` results in any missing values. The special float value `nan` is not considered a missing value.

    Further reading
    ---------------
    See `LD Score regression distinguishes confounding from polygenicity in genome-wide association studies (Bulik-Sullivan et al, 2015) <https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4495769/>`__ 
    and `Partitioning heritability by functional annotation using genome-wide association summary statistcs (Finucane et al, 2015) <https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4626285/>`__ 
    for more in-depth discussion of LD scores.

    Parameters
    ----------
    entry_expr  : :class:`.NumericExpression`
        Expression for entries of genotype matrix (e.g. mt.GT.n_alt_alleles()).
    annotations : :class:`.NumericExpression` or :obj:`list` of :class:`.NumericExpression`
        Expression for annotation(s) to partition LD scores.
    position   : :class:`.NumericExpression`
        Expression for position of variant (e.g. mt.cm_position, mt.locus.position).
    window_size : :obj:`int` or :obj:`float`
        Size of variant window used to calculate LD scores, in units of `position`.

    Returns
    -------
    :class:`.HailTable` 
        Locus-keyed table with LD scores for each variant and annotation.
    """

    mt = entry_expr._indices.source
    annotations = wrap_to_list(annotations)
    variant_key = [x for x in mt.row_key]

    ht_annotations = mt.select_rows(*annotations).rows()
    annotation_names = [x for x in ht_annotations.row if x not in variant_key]

    ht_annotations = hl.Table.union(*[(ht_annotations.annotate(annotation=hl.str(x), value=hl.float(ht_annotations[x]))
                                                     .select('annotation', 'value')) for x in annotation_names])
    mt_annotations = ht_annotations.to_matrix_table(row_key=variant_key, col_key=['annotation'])

    G = BlockMatrix.from_entry_expr(entry_expr)
    A = BlockMatrix.from_entry_expr(mt_annotations.value)

    variant_positions = np.array(position.collect())
    n = G.n_cols

    starts, stops = _compute_row_intervals(variant_positions, window_size)

    R = ((G @ G.T)/n)
    R_squared = R ** 2
    R_squared_adj = R_squared - (1.0 - R_squared)/(n - 2.0)
    R_squared_adj_sparse = R_squared_adj.sparsify_row_intervals(starts=starts, stops=stops)

    L_squared = R_squared_adj_sparse @ A

    tmp_bm_path = new_temp_file()
    tmp_tsv_path = new_temp_file()

    L_squared.write(tmp_bm_path, force_row_major=True)
    BlockMatrix.export(tmp_bm_path, tmp_tsv_path)

    ht_scores = hl.import_table(tmp_tsv_path, no_header=True, impute=True)
    ht_scores = ht_scores.add_index()
    ht_scores = ht_scores.key_by('idx')
    ht_scores = ht_scores.rename({'f{:}'.format(i): annotation_names[i] for i in range(len(annotation_names))})

    ht_variants = mt.rows()
    ht_variants = ht_variants.drop(*[x for x in ht_variants.row if x not in variant_key])
    ht_variants = ht_variants.add_index()
    ht_variants = ht_variants.key_by('idx')

    ht_scores = ht_variants.join(ht_scores, how='inner')
    ht_scores = ht_scores.key_by('locus')
    ht_scores = ht_scores.drop('alleles', 'idx')

    return ht_scores
