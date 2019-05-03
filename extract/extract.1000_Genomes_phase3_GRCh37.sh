#!/bin/bash

URL_ROOT="http://ftp.1000genomes.ebi.ac.uk/vol1/ftp/release/20130502";

for i in {1..22}; do
  wget -c -O - "${URL_ROOT}/ALL.chr${i}.phase3_shapeit2_mvncall_integrated_v5a.20130502.genotypes.vcf.gz" |
  zcat | 
  bgzip -c |
  gsutil cp - gs://hail-datasets-raw-data/1000_Genomes/1000_Genomes_phase3_chr${i}_GRCh37.vcf.bgz
done

wget -c -O - "${URL_ROOT}/ALL.chrX.phase3_shapeit2_mvncall_integrated_v1b.20130502.genotypes.vcf.gz" |
zcat |
bgzip -c |
gsutil cp - gs://hail-datasets-raw-data/1000_Genomes/1000_Genomes_phase3_chrX_GRCh37.vcf.bgz

wget -c -O - "${URL_ROOT}/ALL.chrY.phase3_integrated_v2a.20130502.genotypes.vcf.gz" |
zcat |
bgzip -c |
gsutil cp - gs://hail-datasets-raw-data/1000_Genomes/1000_Genomes_phase3_chrY_GRCh37.vcf.bgz

wget -c -O - "${URL_ROOT}/ALL.chrMT.phase3_callmom-v0_4.20130502.genotypes.vcf.gz" |
zcat |
bgzip -c |
gsutil cp - gs://hail-datasets-raw-data/1000_Genomes/1000_Genomes_phase3_chrMT_GRCh37.vcf.bgz

wget -c -O - "${URL_ROOT}/ALL.wgs.phase3_shapeit2_mvncall_integrated_v5b.20130502.sites.vcf.gz" |
zcat |
bgzip -c |
gsutil cp - gs://hail-datasets-raw-data/1000_Genomes/1000_Genomes_phase3_sites_GRCh37.vcf.bgz

