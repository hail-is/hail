#!/bin/bash

SNVS_URL="http://krishna.gs.washington.edu/download/CADD/v1.4/GRCh38/whole_genome_SNVs.tsv.gz"
INDELS_URL="http://krishna.gs.washington.edu/download/CADD/v1.4/GRCh38/InDels.tsv.gz"

wget -c -O - $SNVS_URL $INDELS_URL | \
zcat | \
grep -v '^#' | \
awk -v FS=$'\t' -v OFS=$'\t' 'BEGIN {print "chromosome","position","ref","alt","raw_score","PHRED_score"} {print $0}' | \
bgzip -c | \
gsutil cp - gs://hail-datasets-raw-data/CADD/CADD_v1.4_GRCh38.tsv.bgz

