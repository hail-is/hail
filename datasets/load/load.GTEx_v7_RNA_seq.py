
import hail as hl
import argparse

raw_data_root = 'gs://hail-datasets-raw-data/GTEx'
hail_data_root = 'gs://hail-datasets-hail-data'

parser = argparse.ArgumentParser()
parser.add_argument('-b', required=True, choices=['GRCh37', 'GRCh38'], help='Ensembl reference genome build.')
args = parser.parse_args()

version = 'v7'
build = args.b

ht_samples = hl.import_table(f'{raw_data_root}/GTEx_v7_sample_attributes.tsv.bgz',
                             impute=True, key='SAMPID', missing='')
ht_samples = ht_samples.rename({'SAMPID': 's'})
ht_samples.write('hdfs:///tmp/samples.ht', overwrite=True)
ht_samples = hl.read_table('hdfs:///tmp/samples.ht')

ht_subjects = hl.import_table(f'{raw_data_root}/GTEx_v7_subject_phenotypes.tsv.bgz',
                              missing='', key='SUBJID')
ht_subjects = ht_subjects.rename({'SUBJID': 'subject_id', 'AGE': 'age_range'})
ht_subjects = ht_subjects.annotate(is_female=ht_subjects['SEX'] == '2',
                                   death_classification_hardy_scale=(hl.case()
                                                                       .when(ht_subjects['DTHHRDY'] == '0', '0_ventilator_case')
                                                                       .when(ht_subjects['DTHHRDY'] == '1', '1_violent_and_fast_death')
                                                                       .when(ht_subjects['DTHHRDY'] == '2', '2_fast_death_of_natural_causes')
                                                                       .when(ht_subjects['DTHHRDY'] == '3', '3_intermediate_death')
                                                                       .when(ht_subjects['DTHHRDY'] == '4', '4_slow_death')
                                                                       .default(hl.missing(hl.tstr))))
ht_subjects = ht_subjects.select('is_female', 'age_range', 'death_classification_hardy_scale')
ht_subjects.write('hdfs:///tmp/subjects.ht', overwrite=True)
ht_subjects = hl.read_table('hdfs:///tmp/subjects.ht')

ht_genes = hl.import_table(f'{raw_data_root}/GTEx_v7_gencode_v19_patched_contigs_genes.gtf.bgz',
                           comment='#', no_header=True, missing='.',
                           types={'f3': hl.tint,
                                  'f4': hl.tint,
                                  'f5': hl.tfloat,
                                  'f7': hl.tint})
ht_genes = ht_genes.rename({'f0': 'seqname',
                            'f1': 'source',
                            'f2': 'feature',
                            'f3': 'start',
                            'f4': 'end',
                            'f5': 'score',
                            'f6': 'strand',
                            'f7': 'frame',
                            'f8': 'attribute'})

ht_genes = ht_genes.annotate(attribute=hl.dict(
    hl.map(lambda x: (x.split(' ')[0],
                      x.split(' ')[1].replace('"', '').replace(';$', '')),
           ht_genes['attribute'].split('; '))))

attributes = ht_genes.aggregate(hl.agg.explode(lambda x: hl.agg.collect_as_set(x), ht_genes['attribute'].keys()))

ht_genes = ht_genes.transmute(**{x: hl.or_missing(ht_genes['attribute'].contains(x),
                                                  ht_genes['attribute'][x]) for x in attributes if x})

ht_genes = ht_genes.annotate(gene_interval=hl.locus_interval(ht_genes['seqname'], ht_genes['start'], ht_genes['end'] + 1, reference_genome='GRCh37'))
ht_genes = ht_genes.filter(ht_genes['feature'] == 'gene')
ht_genes = ht_genes.key_by('gene_id')
ht_genes = ht_genes.select('gene_interval', 'source', 'gene_name', 'havana_gene', 'gene_type', 'gene_status', 'level', 'score', 'strand', 'frame', 'tag')
ht_genes = ht_genes.rename({'gene_name': 'gene_symbol', 'havana_gene': 'havana_gene_id'})
ht_genes.write('hdfs:///tmp/genes.ht', overwrite=True)
ht_genes = hl.read_table('hdfs:///tmp/genes.ht')

# gene read counts
name = 'GTEx_RNA_seq_gene_read_counts'
mt = hl.import_matrix_table(f'{raw_data_root}/GTEx_v7_RNA_seq_gene_read_counts.tsv.bgz',
                            row_fields={'Name': hl.tstr, 'Description': hl.tstr},
                            row_key='Name',
                            entry_type=hl.tstr,
                            missing=' ')
mt = mt.select_entries(read_count=hl.int(hl.float(mt.x)))
mt = mt.rename({'Name': 'gene_id', 'Description': 'gene_symbol', 'col_id': 's'})
mt = mt.annotate_cols(subject_id=hl.delimit(mt['s'].split('-')[:2], '-'))
mt = mt.annotate_cols(**ht_samples[mt.s])
mt = mt.annotate_cols(**ht_subjects[mt.subject_id])
mt = mt.annotate_rows(**ht_genes[mt.gene_id])

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
mt = mt.annotate_cols(subject_id=hl.delimit(mt['s'].split('-')[:2], '-'))
mt = mt.annotate_cols(**ht_samples[mt.s])
mt = mt.annotate_cols(**ht_subjects[mt.subject_id])
mt = mt.annotate_rows(**ht_genes[mt.gene_id])

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

# junction read counts
name = 'GTEx_RNA_seq_junction_read_counts'
mt = hl.import_matrix_table(f'{raw_data_root}/GTEx_v7_RNA_seq_junction_read_counts.tsv.bgz',
                            row_fields={'junction_id': hl.tstr, 'Description': hl.tstr},
                            row_key='junction_id',
                            entry_type=hl.tstr,
                            missing=' ')
mt = mt.select_entries(TPM=hl.int(hl.float(mt.x)))
mt = mt.annotate_rows(_split=mt['junction_id'].split('_'))
mt = mt.annotate_rows(junction_interval=hl.locus_interval(mt['_split'][0], hl.int(mt['_split'][1]), hl.int(mt['_split'][2]) + 1, reference_genome='GRCh37'))
mt = mt.rename({'Description': 'gene_id', 'col_id': 's'})
mt = mt.select_rows('junction_interval', 'gene_id')
mt = mt.annotate_rows(**ht_genes[mt.gene_id])
mt = mt.annotate_cols(subject_id=hl.delimit(mt['s'].split('-')[:2], '-'))
mt = mt.annotate_cols(**ht_samples[mt.s])
mt = mt.annotate_cols(**ht_subjects[mt.s])

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

