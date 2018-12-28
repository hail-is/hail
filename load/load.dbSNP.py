
import argparse
import hail as hl

parser = argparse.ArgumentParser()
parser.add_argument('-v', required=True, choices=['150', '151'], help='dbSNP build to load.')
parser.add_argument('-b', required=True, choices=['GRCh37', 'GRCh38'], help='Reference genome build to load.')
args = parser.parse_args()

name = 'dbSNP'
version = 'build_{}'.format(args.v)
reference_genome = args.b

if reference_genome == 'GRCh37':
    mt = hl.import_vcf('gs://hail-datasets-extracted-data/dbSNP/{n}.{v}.{rg}.vcf.bgz'.format(n=name, v=version, rg=reference_genome),
                       reference_genome=reference_genome)
else:
    contigs = {str(i): 'chr' + str(i) for i in range(1, 23)}
    contigs.update({'X': 'chrX', 'Y': 'chrY', 'MT': 'chrM'})
    mt = hl.import_vcf('gs://hail-datasets-extracted-data/dbSNP/{n}.{v}.{rg}.vcf.bgz'.format(n=name, v=version, rg=reference_genome),
                       reference_genome=reference_genome,
                       contig_recoding=contigs)

mt_split = hl.split_multi(mt)
mt_split = mt_split.annotate_rows(info=mt_split['info'].annotate(CAF=mt_split['info']['CAF'][mt_split['a_index'] - 1],
                                                                 TOPMED=mt_split['info']['TOPMED'][mt_split['a_index'] - 1]))
mt_split = mt_split.drop('old_locus', 'old_alleles')
ht = mt_split.rows()

n_rows = ht.count()
n_partitions = ht.n_partitions()
ht = ht.annotate_globals(metadata=hl.struct(name=name,
                                            version=version,
                                            reference_genome=reference_genome,
                                            n_rows=n_rows,
                                            n_partitions=n_partitions))

ht.describe()
ht.write('gs://hail-datasets/{n}.{v}.{rg}.ht'.format(n=name, v=version, rg=reference_genome), overwrite=True)
