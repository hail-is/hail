
import hail as hl
import argparse

raw_data_root = 'gs://hail-datasets-raw-data/CADD'
hail_data_root = 'gs://hail-datasets-hail-data'

parser = argparse.ArgumentParser()
parser.add_argument('-v', required=True, help='CADD version.')
parser.add_argument('-b', required=True, choices=['GRCh37', 'GRCh38'], help='Ensembl reference genome build.')
args = parser.parse_args()

name = 'CADD'
version = f'v{args.v}'
build = args.b

ht = hl.import_table(f'{raw_data_root}/CADD_{version}_{build}.tsv.bgz',
                     types={'position': hl.tint,
                            'raw_score': hl.tfloat,
                            'PHRED_score': hl.tfloat})

n_rows = ht.count()
n_partitions = ht.n_partitions()

if build == 'GRCh37':
    ht = ht.annotate(locus=hl.locus(ht['chromosome'], ht['position'], build))
else:
    ht = ht.annotate(locus=hl.locus('chr' + ht['chromosome'], ht['position'], build))

ht = ht.annotate(alleles=hl.array([ht['ref'], ht['alt']]))
ht = ht.key_by('locus', 'alleles')
ht = ht.select('raw_score', 'PHRED_score')

ht = ht.annotate_globals(metadata=hl.struct(name=name,
                                            version=version,
                                            reference_genome=build,
                                            n_rows=n_rows,
                                            n_partitions=n_partitions))

ht.write(f'{hail_data_root}/{name}.{version}.{build}.ht', overwrite=True)
ht = hl.read_table(f'{hail_data_root}/{name}.{version}.{build}.ht')
ht.describe()
