
import hail as hl

ht_samples = hl.read_table('gs://hail-datasets-hail-data/1000_Genomes_phase3_samples.ht')
ht_relationships = hl.read_table('gs://hail-datasets-hail-data/1000_Genomes_phase3_sample_relationships.ht')

mt = hl.import_vcf(
    'gs://hail-datasets-raw-data/1000_Genomes/1000_Genomes_phase3_chrMT_GRCh37.vcf.bgz',
    reference_genome='GRCh37')

mt = mt.annotate_cols(**ht_samples[mt.s])
mt = mt.annotate_cols(**ht_relationships[mt.s])

mt_split = hl.split_multi(mt)
mt_split = mt_split.select_entries(GT=hl.downcode(mt_split.GT, mt_split.a_index))
mt_split = mt_split.annotate_rows(
    info=hl.struct(
        AC=mt_split.info.AC[mt_split.a_index - 1],
        VT=(hl.case()
           .when((mt_split.alleles[0].length() == 1) & (mt_split.alleles[1].length() == 1), 'SNP')
           .when(mt_split.alleles[0].matches('<CN*>') | mt_split.alleles[1].matches('<CN*>'), 'SV')
           .default('INDEL'))))

n_rows, n_cols = mt_split.count()
n_partitions = mt_split.n_partitions()

mt_split = hl.sample_qc(mt_split)
mt_split = hl.variant_qc(mt_split)

mt_split = mt_split.annotate_globals(
    metadata=hl.struct(
        name='1000_Genomes_phase3_chrMT',
        reference_genome='GRCh37',
        n_rows=n_rows,
        n_cols=n_cols,
        n_partitions=n_partitions))

mt_split.write('gs://hail-datasets-hail-data/1000_Genomes_phase3_chrMT.GRCh37.mt', overwrite=True)

mt = hl.read_matrix_table('gs://hail-datasets-hail-data/1000_Genomes_phase3_chrMT.GRCh37.mt')
mt.describe()
print(mt.count())
