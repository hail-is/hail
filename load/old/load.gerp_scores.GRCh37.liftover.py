
import hail as hl

name = 'GERP++_scores'
version = None
reference_genome = 'GRCh37'


ht = hl.import_table('gs://hail-datasets/raw-data/GERP/GERP++_scores.hg19.tsv.bgz',
                     types={'position': hl.tint, 'N': hl.tfloat, 'S': hl.tfloat})

hg19 = hl.ReferenceGenome.from_fasta_file('hg19', 
                                          'gs://hail-datasets/raw-data/assemblies/ucsc.hg19.fasta.gz', 
                                          'gs://hail-datasets/raw-data/assemblies/ucsc.hg19.fasta.fai')
hg19.add_liftover('gs://hail-datasets/raw-data/assemblies/hg19tob37.chain.gz', 'GRCh37')

ht = ht.annotate(locus=hl.locus(ht.contig, ht.position, 'hg19'))
ht.write('hdfs:///tmp/tmp.ht', overwrite=True)

ht = hl.read_table('hdfs:///tmp/gerp_scores.hg19.ht')
ht = ht.annotate(locus=hl.liftover(ht.locus, 'GRCh37'))
ht = ht.filter(hl.is_defined(ht.locus), keep=True)
ht = ht.select(ht.locus, ht.N, ht.N)
ht = ht.key_by(ht.locus)

n_rows = ht.count()
n_partitions = ht.n_partitions()

ht = ht.annotate_globals(name=name,
                         version=version,
                         reference_genome=reference_genome,
                         n_rows=n_rows,
                         n_partitions=n_partitions)

ht.describe()
ht.write('gs://hail-datasets/hail-data/{n}.{rg}.ht'.format(n=name, rg=reference_genome), overwrite=True)
