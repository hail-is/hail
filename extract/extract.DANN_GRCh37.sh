#!/bin/bash

wget -c -O - https://cbcl.ics.uci.edu/public_data/DANN/data/DANN_whole_genome_SNVs.tsv.bgz | \
zcat | \
awk -v FS=$'\t' -v OFS=$'\t' 'BEGIN {print "chromosome","position","ref","alt","DANN_score"} {print $0}' | \
bgzip -c | \
gsutil cp - gs://hail-datasets-raw-data/DANN/DANN_GRCh37.tsv.bgz

