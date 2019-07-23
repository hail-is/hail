
import hail as hl
import argparse

raw_data_root = 'gs://hail-datasets-raw-data/Ensembl'
hail_data_root = 'gs://hail-datasets-hail-data'

parser = argparse.ArgumentParser()
parser.add_argument('-v', required=True, help='Dataset version.')
parser.add_argument('-b', required=True, choices=['GRCh37', 'GRCh38'], help='Ensembl reference genome build.')
args = parser.parse_args()

name = 'Ensembl_homo_sapiens_reference_genome'
version = args.v
build = args.b

ht = hl.import_table(f'{raw_data_root}/Ensembl_homo_sapiens_reference_genome_release{version}_{build}.tsv.bgz')

if build == 'GRCh37':
    ht = ht.annotate(locus=hl.locus(ht['chromosome'], hl.int(ht['position']), 'GRCh37'))
else:
    ht = ht.annotate(locus=hl.locus('chr' + ht['chromosome'].replace('MT', 'M'), hl.int(ht['position']), 'GRCh38'))

ht = ht.key_by('locus')
ht = ht.select('reference_allele')

n_rows = ht.count()
n_partitions = ht.n_partitions()

ht = ht.annotate_globals(metadata=hl.struct(name=name,
                                            version=f'release_{version}',
                                            reference_genome=build,
                                            n_rows=n_rows,
                                            n_partitions=n_partitions))

ht.write(f'{hail_data_root}/{name}.release_{version}.{build}.ht', overwrite=True)
ht = hl.read_table(f'{hail_data_root}/{name}.release_{version}.{build}.ht')
ht.describe()
