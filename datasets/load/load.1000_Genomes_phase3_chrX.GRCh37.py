
import hail as hl

ht_samples = hl.read_table('gs://hail-datasets-hail-data/1000_Genomes_phase3_samples.ht')
ht_relationships = hl.read_table('gs://hail-datasets-hail-data/1000_Genomes_phase3_sample_relationships.ht')

mt = hl.import_vcf(
    'gs://hail-datasets-raw-data/1000_Genomes/1000_Genomes_phase3_chrX_GRCh37.vcf.bgz',
    reference_genome='GRCh37')

mt = mt.annotate_cols(**ht_samples[mt.s])
mt = mt.annotate_cols(**ht_relationships[mt.s])

mt_split = hl.split_multi(mt)
mt_split = mt_split.select_entries(GT=hl.downcode(mt_split.GT, mt_split.a_index))
mt_split = mt_split.annotate_rows(info=hl.struct(
    CIEND=mt_split.info.CIEND[mt_split.a_index - 1],
    CIPOS=mt_split.info.CIPOS[mt_split.a_index - 1],
    CS=mt_split.info.CS,
    END=mt_split.info.END,
    IMPRECISE=mt_split.info.IMPRECISE,
    MC=mt_split.info.MC,
    MEINFO=mt_split.info.MEINFO,
    MEND=mt_split.info.MEND,
    MLEN=mt_split.info.MLEN,
    MSTART=mt_split.info.MSTART,
    SVLEN=mt_split.info.SVLEN,
    SVTYPE=mt_split.info.SVTYPE,
    TSD=mt_split.info.TSD,
    AC=mt_split.info.AC[mt_split.a_index - 1],
    AF=mt_split.info.AF[mt_split.a_index - 1],
    NS=mt_split.info.NS,
    AN=mt_split.info.AN,
    EAS_AF=mt_split.info.EAS_AF[mt_split.a_index - 1],
    EUR_AF=mt_split.info.EUR_AF[mt_split.a_index - 1],
    AFR_AF=mt_split.info.AFR_AF[mt_split.a_index - 1],
    AMR_AF=mt_split.info.AMR_AF[mt_split.a_index - 1],
    SAS_AF=mt_split.info.SAS_AF[mt_split.a_index - 1],
    DP=mt_split.info.DP,
    AA=mt_split.info.AA,
    VT=(hl.case()
       .when((mt_split.alleles[0].length() == 1) & (mt_split.alleles[1].length() == 1), 'SNP')
       .when(mt_split.alleles[0].matches('<CN*>') | mt_split.alleles[1].matches('<CN*>'), 'SV')
       .default('INDEL')),
    EX_TARGET=mt_split.info.EX_TARGET,
    MULTI_ALLELIC=mt_split.info.MULTI_ALLELIC))

n_rows, n_cols = mt_split.count()
n_partitions = mt_split.n_partitions()

mt_split = hl.sample_qc(mt_split)
mt_split = hl.variant_qc(mt_split)

mt_split = mt_split.annotate_globals(
    metadata=hl.struct(
        name='1000_Genomes_phase3_chrX',
        reference_genome='GRCh37',
        n_rows=n_rows,
        n_cols=n_cols,
        n_partitions=n_partitions))

mt_split.write('gs://hail-datasets-hail-data/1000_Genomes_phase3_chrX.GRCh37.mt', overwrite=True)

mt = hl.read_matrix_table('gs://hail-datasets-hail-data/1000_Genomes_phase3_chrX.GRCh37.mt')
mt.describe()
print(mt.count())

