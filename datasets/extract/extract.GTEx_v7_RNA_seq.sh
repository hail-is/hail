#!/bin/bash

URL_ROOT="https://storage.googleapis.com/gtex_analysis_v7/rna_seq_data";
BUCKET_ROOT="gs://hail-datasets-raw-data/GTEx"

wget -c -O - ${URL_ROOT}/GTEx_Analysis_2016-01-15_v7_RNASeQCv1.1.8_gene_reads.gct.gz |
zcat |
tail -n+3 |
tr [:blank:] $'\t' |
bgzip -c |
gsutil cp - ${BUCKET_ROOT}/GTEx_v7_RNA_seq_gene_read_counts.tsv.bgz

wget -c -O - ${URL_ROOT}/GTEx_Analysis_2016-01-15_v7_RNASeQCv1.1.8_gene_tpm.gct.gz |
zcat |
tail -n+3 |
tr [:blank:] $'\t' |
bgzip -c |
gsutil cp - ${BUCKET_ROOT}/GTEx_v7_RNA_seq_gene_TPMs.tsv.bgz

wget -c -O - ${URL_ROOT}/GTEx_Analysis_2016-01-15_v7_RNASeQCv1.1.8_gene_median_tpm.gct.gz |
zcat |
tail -n+3 |
tr [:blank:] $'\t' |
bgzip -c |
gsutil cp - ${BUCKET_ROOT}/GTEx_v7_RNA_seq_gene_median_TPMs_per_tissue.tsv.bgz

wget -c -O - ${URL_ROOT}/GTEx_Analysis_2016-01-15_v7_STARv2.4.2a_junctions.gct.gz | 
zcat |
tail -n+3 |
tr [:blank:] $'\t' |
bgzip -c |
gsutil cp - ${BUCKET_ROOT}/GTEx_v7_RNA_seq_junction_read_counts.tsv.bgz

wget -c -O - ${URL_ROOT}/GTEx_Analysis_2016-01-15_v7_RSEMv1.2.22_transcript_expected_count.txt.gz |
zcat |
tr [:blank:] $'\t' |
bgzip -c |
gsutil cp - ${BUCKET_ROOT}/GTEx_v7_RNA_seq_transcript_read_counts.tsv.bgz

wget -c -O - ${URL_ROOT}/GTEx_Analysis_2016-01-15_v7_RSEMv1.2.22_transcript_tpm.txt.gz |
zcat |
tr [:blank:] $'\t' |
bgzip -c |
gsutil cp - ${BUCKET_ROOT}/GTEx_v7_RNA_seq_transcript_TPMs.tsv.bgz

wget -c -O - ${URL_ROOT}/GTEx_Analysis_2016-01-15_v7_RNASeQCv1.1.8_exon_reads.txt.gz |
zcat |
tr [:blank:] $'\t' |
bgzip -c |
gsutil cp - ${BUCKET_ROOT}/GTEx_v7_RNA_seq_exon_read_counts.tsv.bgz

