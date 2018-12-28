
import hail as hl
import argparse

EXTRACT_BUCKET = 'gs://hail-datasets-extracted-data/'
HAIL_BUCKET = 'gs://hail-datasets/'

parser = argparse.ArgumentParser()
parser.add_argument('-v', required=True, help='CADD version.')
parser.add_argument('-b', required=True, choices=['GRCh37', 'GRCh38'], help='Ensembl reference genome build.')
args = parser.parse_args()

name = 'CADD'
version = 'v' + args.v
reference_genome = args.b

ht = hl.import_table(EXTRACT_BUCKET + 'CADD/{n}.{v}.{rg}.tsv.bgz'.format(n=name, v=version, rg=reference_genome),
                     types={'position': hl.tint,
                            'raw_score': hl.tfloat,
                            'PHRED_score': hl.tfloat})
if reference_genome == 'GRCh37':
    ht = ht.annotate(locus=hl.locus(ht['chromosome'], ht['position'], reference_genome))
else:
    ht = ht.annotate(locus=hl.locus('chr' + ht['chromosome'], ht['position'], reference_genome))
ht = ht.annotate(alleles=hl.array([ht['ref'], ht['alt']]))

ht = ht.drop('chromosome', 'position', 'ref', 'alt')
ht = ht.select('locus', 'alleles', 'raw_score', 'PHRED_score')
ht = ht.key_by('locus', 'alleles')

n_rows = ht.count()
n_partitions = ht.n_partitions()
ht = ht.annotate_globals(metadata=hl.struct(name=name,
                                            version=version,
                                            reference_genome=reference_genome,
                                            n_rows=n_rows,
                                            n_partitions=n_partitions))

ht.describe()
ht.write(HAIL_BUCKET + '{n}.{v}.{rg}.ht'.format(n=name, v=version, rg=args.b), overwrite=True)
