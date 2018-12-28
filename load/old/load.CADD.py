
import hail as hl
import argparse

EXTRACT_BUCKET = 'gs://hail-datasets-extracted-data/'
HAIL_BUCKET = 'gs://hail-datasets/'

parser = argparse.ArgumentParser()
parser.add_argument('-v', required=True, help='CADD version.')
parser.add_argument('-b', required=True, choices=['GRCh37', 'GRCh38'], help='Ensembl reference genome build.')
args = parser.parse_args()

name = 'CADD'
version = args.v
reference_genome = args.b

ht = hl.import_table(EXTRACT_BUCKET + 'CADD/{n}.{v}.{rg}.tsv.bgz'.format(n=name, v=version, rg=reference_genome))
ht.describe()
ht.show()
