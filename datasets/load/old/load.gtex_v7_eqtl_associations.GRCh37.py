
import hail as hl

tissues = [
  'Adipose_Subcutaneous',
  'Adipose_Visceral_Omentum',
  'Adrenal_Gland',
  'Artery_Aorta',
  'Artery_Coronary', 
  'Artery_Tibial',
  'Brain_Amygdala', 
  'Brain_Anterior_cingulate_cortex_BA24',
  'Brain_Caudate_basal_ganglia',
  'Brain_Cerebellar_Hemisphere',
  'Brain_Cerebellum',
  'Brain_Cortex',
  'Brain_Frontal_Cortex_BA9',
  'Brain_Hippocampus',
  'Brain_Hypothalamus', 
  'Brain_Nucleus_accumbens_basal_ganglia',
  'Brain_Putamen_basal_ganglia',
  'Brain_Spinal_cord_cervical_c-1',
  'Brain_Substantia_nigra',
  'Breast_Mammary_Tissue',
  'Cells_EBV-transformed_lymphocytes',
  'Cells_Transformed_fibroblasts',
  'Colon_Sigmoid',
  'Colon_Transverse',
  'Esophagus_Gastroesophageal_Junction',
  'Esophagus_Mucosa',
  'Esophagus_Muscularis',
  'Heart_Atrial_Appendage',
  'Heart_Left_Ventricle',
  'Liver',
  'Lung',
  'Minor_Salivary_Gland',
  'Muscle_Skeletal',
  'Nerve_Tibial',
  'Ovary',
  'Pancreas',
  'Pituitary',
  'Prostate',
  'Skin_Not_Sun_Exposed_Suprapubic',
  'Skin_Sun_Exposed_Lower_leg',
  'Small_Intestine_Terminal_Ileum',
  'Spleen',
  'Stomach',
  'Testis',
  'Thyroid',
  'Uterus',
  'Vagina',
  'Whole_Blood'
]

hts = [(hl.import_table('gs://hail-datasets/raw-data/gtex/v7/single-tissue-eqtl/processed/{}.allpairs.tsv.bgz'.format(x))
          .annotate(tissue='{}'.format(x))) for x in tissues]

ht_union = hl.Table.union(*hts)
ht_union = ht_union.annotate(**hl.parse_variant(ht_union.variant_id.replace('_b37$', '').replace('_', ':')))
ht_union = ht_union.drop('variant_id')
ht_union = ht_union.annotate(tss_distance=hl.int(ht_union['tss_distance']),
                             maf=hl.float(ht_union['maf']),
                             ma_samples=hl.int(ht_union['ma_samples']),
                             ma_count=hl.int(ht_union['ma_count']),
                             pval_nominal=hl.float(ht_union['pval_nominal']),
                             slope=hl.float(ht_union['slope']),
                             slope_se=hl.float(hl.or_missing(ht_union['slope_se']!='-nan', ht_union['slope_se'])))

mt = ht_union.to_matrix_table(row_key=['locus', 'alleles', 'gene_id'],
                              col_key=['tissue'],
                              row_fields=['tss_distance', 'maf'])
mt = mt.partition_rows_by(['locus'], 'locus', 'alleles', 'gene_id')

mt.describe()
mt.write('gs://hail-datasets/hail-data/gtex_v7_eqtl_associations.GRCh37.mt', overwrite=True)
