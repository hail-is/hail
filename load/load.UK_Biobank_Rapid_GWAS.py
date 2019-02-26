
import hail as hl

raw_data_root = 'gs://hail-datasets-raw-data/UK_Biobank_Rapid_GWAS'
hail_data_root = 'gs://hail-datasets-hail-data'

for sex in ['both_sexes', 'female', 'male']:

    mt = hl.read_matrix_table(f'{raw_data_root}/gwas.sumstats.{sex}.v2.tmp.mt')

    name = f'UK_Biobank_Rapid_GWAS_{sex}'
    version = 'v2'
    reference_genome = 'GRCh37'

    n_rows, n_cols = mt.count()
    n_partitions = mt.n_partitions()

    mt = mt.annotate_globals(metadata=hl.struct(name=name,
                                                version=version,
                                                reference_genome=reference_genome,
                                                n_rows=n_rows,
                                                n_cols=n_cols,
                                                n_partitions=n_partitions))

    path = f'{hail_data_root}/{name}.{version}.{reference_genome}.mt'
    mt.write(path, overwrite=True)
    mt = hl.read_matrix_table(path)
    mt.describe()
