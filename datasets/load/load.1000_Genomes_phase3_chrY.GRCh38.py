
import hail as hl

ht_samples = hl.read_table('gs://hail-datasets-hail-data/1000_Genomes_phase3_samples.ht')
ht_relationships = hl.read_table('gs://hail-datasets-hail-data/1000_Genomes_phase3_sample_relationships.ht')

mt = hl.import_vcf(
    'gs://hail-datasets-raw-data/1000_Genomes/1000_Genomes_phase3_chrY_GRCh38.vcf.bgz',
    reference_genome='GRCh38',
    contig_recoding={'Y': 'chrY'})

mt = mt.annotate_cols(**ht_samples[mt.s])
mt = mt.annotate_cols(**ht_relationships[mt.s])

mt_split = hl.split_multi(mt)
mt_split = mt_split.select_entries(GT=hl.downcode(mt_split.GT, mt_split.a_index))
mt_split = mt_split.annotate_rows(
    info=hl.struct(
        DP=mt_split.info.DP,
        END=mt_split.info.END,
        SVTYPE=mt_split.info.SVTYPE,
        AA=mt_split.info.AA,
        AC=mt_split.info.AC[mt_split.a_index - 1],
        AF=mt_split.info.AF[mt_split.a_index - 1],
        NS=mt_split.info.NS,
        AN=mt_split.info.AN,
        EAS_AF=mt_split.info.EAS_AF[mt_split.a_index - 1],
        EUR_AF=mt_split.info.EUR_AF[mt_split.a_index - 1],
        AFR_AF=mt_split.info.AFR_AF[mt_split.a_index - 1],
        AMR_AF=mt_split.info.AMR_AF[mt_split.a_index - 1],
        SAS_AF=mt_split.info.SAS_AF[mt_split.a_index - 1],
        VT=(hl.case()
           .when((mt_split.alleles[0].length() == 1) & (mt_split.alleles[1].length() == 1), 'SNP')
           .when(mt_split.alleles[0].matches('<CN*>') | mt_split.alleles[1].matches('<CN*>'), 'SV')
           .default('INDEL')),
        EX_TARGET=mt_split.info.EX_TARGET,
        MULTI_ALLELIC=mt_split.info.MULTI_ALLELIC,
        STRAND_FLIP=mt_split.info.STRAND_FLIP,
        REF_SWITCH=mt_split.info.REF_SWITCH,
        DEPRECATED_RSID=mt_split.info.DEPRECATED_RSID[mt_split.a_index - 1],
        RSID_REMOVED=mt_split.info.RSID_REMOVED[mt_split.a_index - 1],
        GRCH37_38_REF_STRING_MATCH=mt_split.info.GRCH37_38_REF_STRING_MATCH,
        NOT_ALL_RSIDS_STRAND_CHANGE_OR_REF_SWITCH=mt_split.info.NOT_ALL_RSIDS_STRAND_CHANGE_OR_REF_SWITCH,
        GRCH37_POS=mt_split.info.GRCH37_POS,
        GRCH37_REF=mt_split.info.GRCH37_REF,
        ALLELE_TRANSFORM=mt_split.info.ALLELE_TRANSFORM,
        REF_NEW_ALLELE=mt_split.info.REF_NEW_ALLELE,
        CHROM_CHANGE_BETWEEN_ASSEMBLIES=mt_split.info.CHROM_CHANGE_BETWEEN_ASSEMBLIES[mt_split.a_index - 1]))

n_rows, n_cols = mt_split.count()
n_partitions = mt_split.n_partitions()

mt_split = hl.sample_qc(mt_split)
mt_split = hl.variant_qc(mt_split)

mt_split = mt_split.annotate_globals(
    metadata=hl.struct(
        name='1000_Genomes_phase3_chrY',
        reference_genome='GRCh38',
        n_rows=n_rows,
        n_cols=n_cols,
        n_partitions=n_partitions))

mt_split.write('gs://hail-datasets-hail-data/1000_Genomes_phase3_chrY.GRCh38.mt', overwrite=True)

mt = hl.read_matrix_table('gs://hail-datasets-hail-data/1000_Genomes_phase3_chrY.GRCh38.mt')
mt.describe()
print(mt.count())

