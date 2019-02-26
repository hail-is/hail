
import hail as hl
import argparse

raw_data_root = 'gs://hail-datasets-raw-data/GTEx'
hail_data_root = 'gs://hail-datasets-hail-data'

parser = argparse.ArgumentParser()
parser.add_argument('-b', required=True, choices=['GRCh37', 'GRCh38'], help='Ensembl reference genome build.')
args = parser.parse_args()

version = 'v7'
build = args.b

# gene read counts
name = 'GTEx_RNA_seq_gene_read_counts'
mt = hl.import_matrix_table(f'{raw_data_root}/GTEx_v7_RNA_seq_gene_read_counts.tsv.bgz',
                            row_fields={'Name': hl.tstr, 'Description': hl.tstr},
                            row_key='Name',
                            entry_type=hl.tstr,
                            missing=' ')
mt = mt.select_entries(read_count=hl.int(hl.float(mt.x)))
mt = mt.rename({'Name': 'gene_id', 'Description': 'gene_symbol', 'col_id': 's'})

n_rows, n_cols = mt.count()
n_partitions = mt.n_partitions()

mt = mt.annotate_globals(metadata=hl.struct(name=name,
                                            version=version,
                                            reference_genome=build,
                                            n_rows=n_rows,
                                            n_cols=n_cols,
                                            n_partitions=n_partitions))

path = f'{hail_data_root}/{name}.{version}.{build}.mt'
mt.write(path, overwrite=True)
mt = hl.read_matrix_table(path)
mt.describe()

# gene TPMs
name = 'GTEx_RNA_seq_gene_TPMs'
mt = hl.import_matrix_table(f'{raw_data_root}/GTEx_v7_RNA_seq_gene_TPMs.tsv.bgz',
                            row_fields={'Name': hl.tstr, 'Description': hl.tstr},
                            row_key='Name',
                            entry_type=hl.tstr,
                            missing=' ')
mt = mt.select_entries(TPM=hl.float(mt.x))
mt = mt.rename({'Name': 'gene_id', 'Description': 'gene_symbol', 'col_id': 's'})

n_rows, n_cols = mt.count()
n_partitions = mt.n_partitions()

mt = mt.annotate_globals(metadata=hl.struct(name=name,
                                            version=version,
                                            reference_genome=build,
                                            n_rows=n_rows,
                                            n_cols=n_cols,
                                            n_partitions=n_partitions))

path = f'{hail_data_root}/{name}.{version}.{build}.mt'
mt.write(path, overwrite=True)
mt = hl.read_matrix_table(path)
mt.describe()
