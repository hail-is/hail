
import hail as hl

mt = hl.read_matrix_table('gs://hail-datasets/hail-data/gtex_v7_junction_read_counts.GRCh37.mt')

b37 = hl.get_reference('GRCh37')
b37.add_liftover('gs://hail-common/references/grch37_to_grch38.over.chain.gz', 'GRCh38')

mt = mt.annotate_rows(liftover_junction_interval=hl.liftover(mt.junction_interval, 'GRCh38'),
                      gene_interval=hl.liftover(mt.gene_interval, 'GRCh38'))
mt = mt.filter_rows(hl.is_defined(mt.liftover_junction_interval), keep=True)
mt = mt.partition_rows_by(['liftover_junction_interval'], 'liftover_junction_interval')
mt = mt.drop(mt.junction_interval)
mt = mt.rename({'liftover_junction_interval': 'junction_interval'})

mt.describe()
mt.write('gs://hail-datasets/hail-data/gtex_v7_junction_read_counts.GRCh38.liftover.mt', overwrite=True)
