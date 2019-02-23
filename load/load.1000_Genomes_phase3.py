
import hail as hl
import argparse

raw_data_root = 'gs://hail-datasets-raw-data/1000_Genomes'
hail_data_root = 'gs://hail-datasets-hail-data'

parser = argparse.ArgumentParser()
parser.add_argument('-b', required=True, choices=['GRCh37', 'GRCh38'], help='Ensembl reference genome build.')
args = parser.parse_args()

name = '1000_Genomes_autosomes'
version = 'phase_3'
build = args.b

ht_samples = hl.import_table(f'{raw_data_root}/1000_Genomes_phase3_samples.tsv.bgz')
ht_samples = ht_samples.annotate(is_female=ht_samples['gender'] == 'female')
ht_samples = ht_samples.rename({'sample': 's',
                                'pop': 'population',
                                'super_pop': 'super_population'})
ht_samples = ht_samples.key_by('s')
ht_samples = ht_samples.select('population', 'super_population', 'is_female')
ht_samples.write('hdfs:///tmp/samples.ht', overwrite=True)
ht_samples = hl.read_table('hdfs:///tmp/samples.ht')

ht_relationships = hl.import_table(f'{raw_data_root}/1000_Genomes_phase3_sample_relationships.tsv.bgz')
ht_relationships = ht_relationships.rename({'Family ID': 'family_id',
                                            'Individual ID': 's',
                                            'Paternal ID': 'paternal_id',
                                            'Maternal ID': 'maternal_id',
                                            'Relationship': 'relationship_role',
                                            'Siblings': 'sibling_ids',
                                            'Second Order': 'second_order_relationship_ids',
                                            'Third Order': 'third_order_relationship_ids',
                                            'Children': 'children_ids'})
ht_relationships = ht_relationships.annotate(paternal_id=hl.or_missing(ht_relationships['paternal_id'] != '0',
                                                                       ht_relationships['paternal_id']),
                                             maternal_id=hl.or_missing(ht_relationships['maternal_id'] != '0',
                                                                       ht_relationships['maternal_id']),
                                             relationship_role=hl.cond(ht_relationships['relationship_role'] == 'unrel',
                                                                       'unrelated',
                                                                       ht_relationships['relationship_role']),
                                             sibling_ids=hl.or_missing(ht_relationships['sibling_ids'] == '0',
                                                                       hl.map(lambda x: x.strip(), ht_relationships['sibling_ids'].split(','))),
                                             children_ids=hl.or_missing(ht_relationships['children_ids'] == '0',
                                                                        hl.map(lambda x: x.strip(), ht_relationships['children_ids'].split(','))),
                                             second_order_relationship_ids=hl.or_missing(ht_relationships['second_order_relationship_ids'] == '0',
                                                                                         hl.map(lambda x: x.strip(), ht_relationships['second_order_relationship_ids'].split(','))),
                                             third_order_relationship_ids=hl.or_missing(ht_relationships['third_order_relationship_ids'] == '0',
                                                                                        hl.map(lambda x: x.strip(), ht_relationships['third_order_relationship_ids'].split(','))))
ht_relationships = ht_relationships.key_by('s')
ht_relationships = ht_relationships.select('family_id',
                                           'relationship_role',
                                           'maternal_id',
                                           'paternal_id',
                                           'children_ids',
                                           'sibling_ids',
                                           'second_order_relationship_ids',
                                           'third_order_relationship_ids')
ht_relationships.write('hdfs:///tmp/relationships.ht', overwrite=True)
ht_relationships = hl.read_table('hdfs:///tmp/relationships.ht')

mt = hl.import_vcf(f'{raw_data_root}/1000_Genomes_phase3_chr*_{build}.vcf.bgz')
mt_split = hl.split_multi(mt)
mt_split = mt_split.select_entries(GT=hl.downcode(mt_split.GT, mt_split.a_index))
mt_split = mt_split.annotate_rows(info=hl.struct(CIEND=mt_split.info.CIEND[mt_split.a_index - 1],
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
mt_split = mt_split.drop('old_locus', 'old_alleles', 'a_index')
mt_split = mt_split.annotate_cols(**ht_samples[mt.s])
mt_split = mt_split.annotate_cols(**ht_relationships[mt.s])
mt_split = hl.sample_qc(mt_split)
mt_split = hl.variant_qc(mt_split)

mt_split.write(f'{hail_data_root}/{name}.{version}.{build}.mt', overwrite=True)
mt = hl.read_matrix_table(f'{hail_data_root}/{name}.{version}.{build}.mt')
mt.describe()
