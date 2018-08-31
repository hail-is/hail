
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

hts = [(hl.import_table('gs://hail-datasets/raw-data/gtex/v7/single-tissue-eqtl/processed/{}.v7.egenes.tsv.bgz'.format(x), impute=True)
          .annotate(tissue='{}'.format(x))) for x in tissues]

ht_union = hl.Table.union(*hts)
ht_union = ht_union.rename({'rs_id_dbSNP147_GRCh37p13': 'rsid'})
ht_union = ht_union.annotate(locus=hl.locus(ht_union['chr'], ht_union['pos']),
                             alleles=hl.array([ht_union['ref'], ht_union['alt']]),
                             gene_interval=hl.interval(hl.locus(ht_union['gene_chr'], ht_union['gene_start'], 'GRCh37'),
                                                       hl.locus(ht_union['gene_chr'], ht_union['gene_end'] + 1, 'GRCh37')))
ht_union = ht_union.drop('variant_id', 'gene_chr', 'gene_start', 'gene_end', 'chr', 'pos', 'ref', 'alt')

mt = ht_union.to_matrix_table(row_key=['locus', 'alleles', 'gene_id'],
                              col_key=['tissue'],
                              row_fields=['rsid', 'gene_interval', 'gene_name', 'strand', 'tss_distance', 'maf'])
mt = mt.partition_rows_by(['locus'], 'locus', 'alleles', 'gene_id')

mt.describe()
mt.write('gs://hail-datasets/hail-data/gtex_v7_eqtl_egenes.GRCh37.mt', overwrite=True)
