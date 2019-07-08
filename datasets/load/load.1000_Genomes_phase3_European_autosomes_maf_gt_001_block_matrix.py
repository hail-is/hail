
import hail as hl
from hail.linalg import BlockMatrix

mt = hl.read_matrix_table('gs://hail-datasets-hail-data/1000_Genomes_phase3_autosomes.GRCh37.mt')

mt = mt.filter_cols(mt.super_population == 'EUR')
mt = hl.variant_qc(mt)
mt = mt.filter_rows((mt.variant_qc.AF[0] > 0.001) & (mt.variant_qc.AF[1] > 0.001))

BlockMatrix.write_from_entry_expr(
    entry_expr=mt.GT.n_alt_alleles(),
    path='gs://hail-datasets-hail-data/1000_Genomes_phase3_European_autosomes_maf_gt_001.bm',
    mean_impute=True,
    center=False,
    normalize=False,
    block_size=4096,
    overwrite=True)

bm = BlockMatrix.read('gs://hail-datasets-hail-data/1000_Genomes_phase3_European_autosomes_maf_gt_001.bm')

metadata = hl.struct(
    name='1000_Genomes_phase3_European_autosomes_maf_gt_001',
    reference_genome='GRCh37',
    n_rows=bm.n_rows,
    n_cols=bm.n_cols,
    block_size=bm.block_size)

hl.experimental.write_expression(
    metadata, 'gs://hail-datasets-hail-data/1000_Genomes_phase3_European_autosomes_maf_gt_001.metadata.he', overwrite=True)
