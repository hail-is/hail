#!/bin/bash

URL_ROOT="https://storage.googleapis.com/gtex_analysis_v7"
BUCKET_ROOT="gs://hail-datasets-raw-data/GTEx"

wget -c -O - ${URL_ROOT}/annotations/GTEx_v7_Annotations_SampleAttributesDS.txt |
bgzip -c |
gsutil cp - ${BUCKET_ROOT}/GTEx_v7_sample_attributes.tsv.bgz 

wget -c -O - ${URL_ROOT}/annotations/GTEx_v7_Annotations_SubjectPhenotypesDS.txt |
bgzip -c |
gsutil cp - ${BUCKET_ROOT}/GTEx_v7_subject_phenotypes.tsv.bgz

wget -c -O - ${URL_ROOT}/reference/GTEx_Analysis_2016-01-15_v7_WholeGenomeSeq_635Ind_PASS_AB02_GQ20_HETX_MISS15_PLINKQC.lookup_table.txt.gz |
zcat |
bgzip -c |
gsutil cp - ${BUCKET_ROOT}/GTEx_v7_variant_annotations.tsv.bgz

wget -c -O - ${URL_ROOT}/reference/gencode.v19.genes.v7.patched_contigs.gtf |
bgzip -c |
gsutil cp - ${BUCKET_ROOT}/GTEx_v7_gencode_v19_patched_contigs_genes.gtf.bgz

wget -c -O - ${URL_ROOT}/reference/gencode.v19.genes.v7.patched_contigs.exons.txt |
bgzip -c |
gsutil cp - ${BUCKET_ROOT}/GTEx_v7_gencode_v19_patched_contigs_exons.tsv.bgz

wget -c -O - ${URL_ROOT}/reference/gencode.v19.transcripts.patched_contigs.gtf |
bgzip -c | 
gsutil cp - ${BUCKET_ROOT}/GTEx_v7_gencode_v19_patched_contigs_transcripts.gtf.bgz

