#!/bin/bash

URL_ROOT="http://ftp.1000genomes.ebi.ac.uk/vol1/ftp/release/20130502/supporting/GRCh38_positions";

for i in {1..22} {X,Y}; do
  wget -c -O - "${URL_ROOT}/ALL.chr${i}_GRCh38.genotypes.20170504.vcf.gz" |
  zcat |
  bgzip -c |
  gsutil cp - gs://hail-datasets-raw-data/1000_Genomes/1000_Genomes_phase3_chr${i}_GRCh38.vcf.bgz

  wget -c -O - "${URL_ROOT}/ALL.chr1_GRCh38_sites.20170504.vcf.gz" | 
  zcat |
  bgzip -c |
  gsutil cp - gs://hail-datasets-raw-data/1000_Genomes/1000_Genomes_phase3_chr${i}_sites_GRCh38.vcf.bgz
done

