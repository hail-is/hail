
import hail as hl

mt = hl.read_matrix_table('gs://hail-datasets/hail-data/1000_genomes_phase3_autosomes.GRCh37.mt')

b37 = hl.get_reference('GRCh37')
b37.add_liftover('gs://hail-common/references/grch37_to_grch38.over.chain.gz', 'GRCh38')

mt = mt.annotate_rows(liftover_locus=hl.liftover(mt.locus, 'GRCh38'))
mt = mt.filter_rows(hl.is_defined(mt.liftover_locus), keep=True)
mt = mt.partition_rows_by(['liftover_locus'], 'liftover_locus', 'alleles')
mt = mt.drop(mt.locus)
mt = mt.rename({'liftover_locus': 'locus'})

mt.describe()
mt = mt.repartition(500)
mt.write('gs://hail-datasets/hail-data/1000_genomes_phase3_autosomes.GRCh38.liftover.mt', overwrite=True)
