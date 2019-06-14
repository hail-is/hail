#!/bin/bash

TISSUES=(
  "Adipose_Subcutaneous"
  "Adipose_Visceral_Omentum"
  "Adrenal_Gland"
  "Artery_Aorta"
  "Artery_Coronary"
  "Artery_Tibial"
  "Brain_Amygdala"
  "Brain_Anterior_cingulate_cortex_BA24"
  "Brain_Caudate_basal_ganglia"
  "Brain_Cerebellar_Hemisphere"
  "Brain_Cerebellum"
  "Brain_Cortex"
  "Brain_Frontal_Cortex_BA9"
  "Brain_Hippocampus"
  "Brain_Hypothalamus"
  "Brain_Nucleus_accumbens_basal_ganglia"
  "Brain_Putamen_basal_ganglia"
  "Brain_Spinal_cord_cervical_c-1"
  "Brain_Substantia_nigra"
  "Breast_Mammary_Tissue"
  "Cells_EBV-transformed_lymphocytes"
  "Cells_Transformed_fibroblasts"
  "Colon_Sigmoid"
  "Colon_Transverse"
  "Esophagus_Gastroesophageal_Junction"
  "Esophagus_Mucosa"
  "Esophagus_Muscularis"
  "Heart_Atrial_Appendage"
  "Heart_Left_Ventricle"
  "Liver"
  "Lung"
  "Minor_Salivary_Gland"
  "Muscle_Skeletal"
  "Nerve_Tibial"
  "Ovary"
  "Pancreas"
  "Pituitary"
  "Prostate"
  "Skin_Not_Sun_Exposed_Suprapubic"
  "Skin_Sun_Exposed_Lower_leg"
  "Small_Intestine_Terminal_Ileum"
  "Spleen"
  "Stomach"
  "Testis"
  "Thyroid"
  "Uterus"
  "Vagina"
  "Whole_Blood"
)

URL_ROOT="https://storage.googleapis.com/gtex_analysis_v7/single_tissue_eqtl_data"
BUCKET_ROOT="gs://hail-datasets-raw-data/GTEx"

wget -c -O /tmp/GTEx_Analysis_v7_eQTL_covariates.tar.gz ${URL_ROOT}/GTEx_Analysis_v7_eQTL_covariates.tar.gz
tar -xzvf /tmp/GTEx_Analysis_v7_eQTL_covariates.tar.gz -C /tmp/

wget -c -O /tmp/GTEx_Analysis_v7_eQTL_expression_matrices.tar.gz ${URL_ROOT}/GTEx_Analysis_v7_eQTL_expression_matrices.tar.gz
tar -xzvf /tmp/GTEx_Analysis_v7_eQTL_expression_matrices.tar.gz -C /tmp/

wget -c -O /tmp/GTEx_Analysis_v7_eQTL.tar.gz ${URL_ROOT}/GTEx_Analysis_v7_eQTL.tar.gz
tar -xzvf /tmp/GTEx_Analysis_v7_eQTL.tar.gz -C /tmp/

for TISSUE in ${TISSUES[@]}; do
  awk -v t=${TISSUE} '
    {
      if (NR==1) {
        print "tissue",$0;
      } else {
        print t,$0;
      }
    }' /tmp/GTEx_Analysis_v7_eQTL_covariates/${TISSUE}.v7.covariates.txt |
  tr [:blank:] $'\t' |
  bgzip -c |
  gsutil cp - ${BUCKET_ROOT}/GTEx_v7_eQTL_covariates_${TISSUE}.tsv.bgz
  
  zcat /tmp/GTEx_Analysis_v7_eQTL_expression_matrices/${TISSUE}.v7.normalized_expression.bed.gz |
  cut -f1,2,4- |
  awk -v t=${TISSUE} '
    {
      if (NR==1) {
        print "tissue",$0;
      } else {
        print t,$0;
      }
    }' |
  tr [:blank:] $'\t' |
  bgzip -c |
  gsutil cp - ${BUCKET_ROOT}/GTEx_v7_eQTL_expression_matrix_${TISSUE}.tsv.bgz

  zcat /tmp/GTEx_Analysis_v7_eQTL/${TISSUE}.v7.egenes.txt.gz |
  awk -v t=${TISSUE} '
    {
      if (NR==1) {
        print "tissue",$0;
      } else {
        print t,$0;
      }
    }' |
  tr [:blank:] $'\t' |
  bgzip -c |
  gsutil cp - ${BUCKET_ROOT}/GTEx_v7_eQTL_egene_associations_${TISSUE}.tsv.bgz 

  zcat /tmp/GTEx_Analysis_v7_eQTL/${TISSUE}.v7.signif_variant_gene_pairs.txt.gz |
  awk -v t=${TISSUE} '
    {
      if (NR==1) {
        print "tissue",$0;
      } else {
        print t,$0;
      }
    }' |
  tr [:blank:] $'\t' |
  bgzip -c |
  gsutil cp - ${BUCKET_ROOT}/GTEx_v7_eQTL_significant_variant_gene_associations_${TISSUE}.tsv.bgz
  
  wget -c -O - ${URL_ROOT}/all_snp_gene_associations/${TISSUE}.allpairs.txt.gz |
  zcat |
  awk -v t=$TISSUE '
    {
      if (NR==1) { 
        print "tissue",$0;
      } else {
        print t,$0;
      }
    }' |
  tr [:blank:] $'\t' |
  bgzip -c |
  gsutil cp - ${BUCKET_ROOT}/GTEx_v7_eQTL_associations_${TISSUE}.tsv.bgz

done

