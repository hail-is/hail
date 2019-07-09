
import hail as hl

ht = hl.read_table('gs://hail-datasets/hail-data/gerp_scores.GRCh37.liftover.ht')

b37 = hl.get_reference('GRCh37')
b37.add_liftover('gs://hail-common/references/grch37_to_grch38.over.chain.gz', 'GRCh38')

ht = ht.annotate(liftover_locus=hl.liftover(ht.locus, 'GRCh38'))
ht = ht.filter(hl.is_defined(ht.liftover_locus), keep=True)
ht = ht.key_by(ht.liftover_locus)
ht = ht.drop('locus')
ht = ht.rename({'liftover_locus': 'locus'})

ht.describe()
ht.write('gs://hail-datasets/hail-data/gerp_scores.GRCh38.liftover.ht', overwrite=True)
