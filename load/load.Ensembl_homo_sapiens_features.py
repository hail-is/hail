
import hail as hl
import argparse

raw_data_root = 'gs://hail-datasets-raw-data/Ensembl'
hail_data_root = 'gs://hail-datasets-hail-data'

parser = argparse.ArgumentParser()
parser.add_argument('-v', required=True, help='Dataset version.')
parser.add_argument('-b', required=True, choices=['GRCh37', 'GRCh38'], help='Ensembl reference genome build.')
args = parser.parse_args()

name = 'Ensembl_homo_sapiens_low_complexity_regions'
version = args.v
build = args.b

ht = hl.import_table(f'{raw_data_root}/Ensembl_homo_sapiens_features_{version}_{build}.gff3.bgz')
ht.describe()
