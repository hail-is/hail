
import hail as hl

ht = hl.import_table('gs://hail-datasets/raw-data/gerp/gerp.scores.hg19.tsv.bgz',
                     types={'position': hl.tint, 'neutral_rate': hl.tfloat, 'RS_score': hl.tfloat})

hg19 = hl.ReferenceGenome.from_fasta_file('hg19', 'gs://hail-datasets/raw-data/assemblies/ucsc.hg19.fasta.gz', 'gs://hail-datasets/raw-data/assemblies/ucsc.hg19.fasta.fai')
hg19.add_liftover('gs://hail-datasets/raw-data/assemblies/hg19tob37.chain.gz', 'GRCh37')

ht = ht.annotate(locus=hl.locus(ht.contig, ht.position, 'hg19'))
ht.write('hdfs:///tmp/gerp_scores.hg19.ht', overwrite=True)

ht = hl.read_table('hdfs:///tmp/gerp_scores.hg19.ht')
ht = ht.annotate(locus=hl.liftover(ht.locus, 'GRCh37'))
ht = ht.filter(hl.is_defined(ht.locus), keep=True)

ht = ht.select(ht.locus, ht.neutral_rate, ht.RS_score)
ht = ht.key_by(ht.locus)

ht.describe()
ht.write('gs://hail-datasets/hail-data/gerp_scores.GRCh37.liftover.ht', overwrite=True)

print(ht.count())
