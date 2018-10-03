
import hail as hl

ht_samples = hl.import_table('gs://hail-datasets/raw-data/1000_genomes/samples_1kg.tsv',
                             key='sample')

mt = hl.import_vcf('gs://hail-datasets/raw-data/1000_genomes/ALL.chrMT.phase3_callmom-v0_4.20130502.genotypes.vcf.bgz')
mt = mt.key_rows_by('locus')
mt = mt.distinct_by_row()
mt = mt.partition_rows_by(['locus'], 'locus', 'alleles')

mt_split = hl.split_multi(mt)
mt_split = mt_split.select_entries(GT=hl.downcode(mt_split.GT, mt_split.a_index))
mt_split = mt_split.annotate_rows(info=hl.struct(AC=mt_split.info.AC,
                                                 VT=(hl.case()
                                                       .when((mt_split.alleles[0].length() == 1) & (mt_split.alleles[1].length() == 1), 'SNP')
                                                       .when(mt_split.alleles[0].matches('<CN*>') | mt_split.alleles[1].matches('<CN*>'), 'SV')
                                                       .default('INDEL'))))
mt_split.describe()
mt_split = mt_split.drop('old_locus', 'old_alleles', 'a_index')

mt_split = mt_split.annotate_cols(sex=ht_samples[mt_split.s].gender,
                                  super_population=ht_samples[mt_split.s].super_pop,
                                  population=ht_samples[mt_split.s].pop)

mt_split = hl.sample_qc(mt_split)
mt_split = hl.variant_qc(mt_split)
mt_split = hl.vep(mt_split, 'gs://hail-common/vep/vep/vep85-gcloud.json')

mt_split.describe()
mt_split.write('gs://hail-datasets/hail-data/1000_genomes_phase3_chrMT.GRCh37.mt', overwrite=True)
