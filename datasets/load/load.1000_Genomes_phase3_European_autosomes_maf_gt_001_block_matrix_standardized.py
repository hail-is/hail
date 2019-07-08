
import hail as hl
from hail.linalg import BlockMatrix

g = BlockMatrix.read('gs://hail-datasets-hail-data/1000_Genomes_phase3_European_autosomes_maf_gt_001.bm')

n = g.shape[1]
m1 = g.sum(axis=1).cache()
m2 = (g**2).sum(axis=1).cache()

mean = m1 / n
stdev = ((m2-m1**2 / n) / (n-1)).sqrt()
g_std = ((g - mean) / stdev)

g_std.write(
    'gs://hail-datasets-hail-data/1000_Genomes_phase3_European_autosomes_maf_gt_001_standardized.bm',
    overwrite=True)

bm = BlockMatrix.read('gs://hail-datasets-hail-data/1000_Genomes_phase3_European_autosomes_maf_gt_001_standardized.bm')

metadata = hl.struct(
    name='1000_Genomes_phase3_European_autosomes_maf_gt_001_standardized',
    reference_genome='GRCh37',
    n_rows=bm.n_rows,
    n_cols=bm.n_cols,
    block_size=bm.block_size)

hl.experimental.write_expression(
    metadata, 'gs://hail-datasets-hail-data/1000_Genomes_phase3_European_autosomes_maf_gt_001_standardized.metadata.he', overwrite=True)
