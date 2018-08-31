
import hail as hl

ht_samples = hl.import_table('gs://hail-datasets/raw-data/1000_genomes/samples_1kg.tsv',
                             key='sample')

ht_map = hl.import_table('gs://hail-datasets/raw-data/1000_genomes/genetic_map_b37.tsv', 
                         types={'bp_position': hl.tint, 'cm_position': hl.tfloat, 'recombination_rate': hl.tfloat})
ht_map = ht_map.annotate(locus=hl.locus(ht_map['chr'], ht_map['bp_position'], 'GRCh37'))
ht_map = ht_map.key_by('locus')

mt = hl.import_vcf('gs://hail-datasets/raw-data/1000_genomes/ALL.chrX.phase3_shapeit2_mvncall_integrated_v1b.20130502.genotypes.vcf.bgz')
mt = mt.key_rows_by('locus')
mt = mt.distinct_by_row()
mt = mt.partition_rows_by(['locus'], 'locus', 'alleles')

mt_split = hl.split_multi(mt)
mt_split = mt_split.select_entries(GT=hl.downcode(mt_split.GT, mt_split.a_index))
mt_split = mt_split.annotate_rows(info=hl.struct(CIEND=mt_split.info.CIEND,
                                                 CIPOS=mt_split.info.CIPOS,
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
                                                 OLD_VARIANT=mt_split.info.OLD_VARIANT,
                                                 VT=(hl.case()
                                                       .when((mt_split.alleles[0].length() == 1) & (mt_split.alleles[1].length() == 1), 'SNP')
                                                       .when(mt_split.alleles[0].matches('<CN*>') | mt_split.alleles[1].matches('<CN*>'), 'SV')
                                                       .default('INDEL')),
                                                 EX_TARGET=mt_split.info.EX_TARGET,
                                                 MULTI_ALLELIC=mt_split.info.MULTI_ALLELIC))
mt_split.describe()

mt_split = mt_split.drop('old_locus', 'old_alleles', 'a_index')
mt_split = mt_split.annotate_cols(sex=ht_samples[mt_split.s].gender,
                                  super_population=ht_samples[mt_split.s].super_pop,
                                  population=ht_samples[mt_split.s].pop)
mt_split = mt_split.annotate_rows(cm_position=ht_map[mt_split.locus].cm_position,
                                  recombination_rate_cm_per_mb=ht_map[mt_split.locus].recombination_rate)

mt_split = hl.sample_qc(mt_split)
mt_split = hl.variant_qc(mt_split)
mt_split = hl.vep(mt_split, 'gs://hail-common/vep/vep/vep85-gcloud.json')

mt_split.describe()
mt_split = mt_split.repartition(50)
mt_split.write('gs://hail-datasets/hail-data/1000_genomes_phase3_chrX.GRCh37.mt', overwrite=True)
