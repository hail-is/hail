#!/bin/bash

while getopts ":b:" args; do
  case $args in
    b)
      case $OPTARG in
        GRCh37|GRCh38)
          BUILD=$OPTARG
          ;;
        *)
          echo "Invalid reference genome build: -$OPTARG takes either \"GRCh37\" or \"GRCh38\" as an argument." >&2
          exit 1
          ;;
      esac
      ;;
    \?)
      echo "Invalid option: -$OPTARG" >&2
      exit 1
      ;;
    :)
      echo "Option -$OPTARG requires an argument." >&2
      exit 1
      ;;
  esac
done

URL_ROOT="http://ftp.1000genomes.ebi.ac.uk/vol1/ftp/release/20130502";

wget -c -O - "${URL_ROOT}/integrated_call_samples_v3.20130502.ALL.panel" |
bgzip -c |
gsutil cp - gs://hail-datasets-raw-data/1000_Genomes/1000_Genomes_phase3_samples.tsv.bgz

wget -c -O - "${URL_ROOT}/integrated_call_samples_v2.20130502.ALL.ped" |
bgzip -c |
gsutil cp - gs://hail-datasets-raw-data/1000_Genomes/1000_Genomes_phase3_sample_relationships.tsv.bgz

if [ $BUILD == "GRCh37" ]; then
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
else
  URL_ROOT="${URL_ROOT}/supporting/GRCh38_positions";

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
fi

