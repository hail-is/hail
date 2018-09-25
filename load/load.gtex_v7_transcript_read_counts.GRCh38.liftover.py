
import hail as hl

mt = hl.read_matrix_table('gs://hail-datasets/hail-data/gtex_v7_transcript_read_counts.GRCh37.mt')

b37 = hl.get_reference('GRCh37')
b37.add_liftover('gs://hail-common/references/grch37_to_grch38.over.chain.gz', 'GRCh38')
mt = mt.annotate_rows(interval=hl.liftover(mt.interval, 'GRCh38'))

mt.describe()
mt.write('gs://hail-datasets/hail-data/gtex_v7_transcript_read_counts.GRCh38.liftover.mt', overwrite=True)
