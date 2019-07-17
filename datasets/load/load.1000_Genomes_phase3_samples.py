
import hail as hl

ht_samples = hl.import_table(
    'gs://hail-datasets-raw-data/1000_Genomes/1000_Genomes_phase3_samples.tsv.bgz',
    no_header=True, filter='sample', find_replace=('\t\t', ''))

ht_samples = ht_samples.annotate(is_female=ht_samples['f3'] == 'female')
ht_samples = ht_samples.rename({'f0': 's',
                                'f1': 'population',
                                'f2': 'super_population'})
ht_samples = ht_samples.key_by('s')
ht_samples = ht_samples.select('population', 'super_population', 'is_female')

n_rows = ht_samples.count()
n_partitions = ht_samples.n_partitions()

ht_samples = ht_samples.annotate_globals(
    metadata=hl.struct(
        name='1000_Genomes_phase3_samples',
        n_rows=n_rows,
        n_partitions=n_partitions))

ht_samples.write('gs://hail-datasets-hail-data/1000_Genomes_phase3_samples.ht', overwrite=True)
