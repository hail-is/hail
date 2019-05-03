#!/bin/bash

wget -O /tmp/1000G_Phase3_baseline_v1.1_ldscores.tgz https://data.broadinstitute.org/alkesgroup/LDSCORE/1000G_Phase3_baseline_v1.1_ldscores.tgz
tar -C /tmp -xzvf /tmp/1000G_Phase3_baseline_v1.1_ldscores.tgz

zcat /tmp/baseline_v1.1/baseline.1.l2.ldscore.gz > /tmp/baseline_v1.1/ld_scores.GRCh37.tsv
for i in {2..22}; do
    zcat /tmp/baseline_v1.1/baseline.${i}.l2.ldscore.gz | tail -n+2 >> /tmp/baseline_v1.1/ld_scores.GRCh37.tsv;
done
bgzip -c /tmp/baseline_v1.1/ld_scores.GRCh37.tsv | gsutil cp - gs://hail-datasets-raw-data/LDSC/baseline_v1.1/ld_scores.GRCh37.tsv.bgz

zcat /tmp/baseline_v1.1/baseline.1.annot.gz > /tmp/baseline_v1.1/annotations.GRCh37.tsv
for i in {2..22}; do
    zcat /tmp/baseline_v1.1/baseline.${i}.annot.gz | tail -n+2 >> /tmp/baseline_v1.1/annotations.GRCh37.tsv;
done
bgzip -c /tmp/baseline_v1.1/annotations.GRCh37.tsv | gsutil cp - gs://hail-datasets-raw-data/LDSC/baseline_v1.1/annotations.GRCh37.tsv.bgz

zcat /tmp/baseline_v1.1/baseline.1.l2.ldscore.gz | head -1 | cut -f4- | tr '\t' '\n' | awk 'BEGIN {print "annotation"} {print}' > /tmp/baseline_v1.1/header.txt
awk '{for (i=1; i<=NF; i++) { sums[i]+=$i; } } END {print "M_5_50"; for (c in sums) { print sums[c]; }}' /tmp/baseline_v1.1/baseline.*.l2.M_5_50 > /tmp/baseline_v1.1/M_5_50.GRCh37.txt
paste /tmp/baseline_v1.1/header.txt /tmp/baseline_v1.1/M_5_50.GRCh37.txt | bgzip -c | gsutil cp - gs://hail-datasets-raw-data/LDSC/baseline_v1.1/M_5_50.GRCh37.tsv.bgz

awk '{for (i=1; i<=NF; i++) { sums[i]+=$i; } } END {print "M"; for (c in sums) { print sums[c]; }}' /tmp/baseline_v1.1/baseline.*.l2.M > /tmp/baseline_v1.1/M.GRCh37.txt
paste /tmp/baseline_v1.1/header.txt /tmp/baseline_v1.1/M.GRCh37.txt | bgzip -c | gsutil cp - gs://hail-datasets-raw-data/LDSC/baseline_v1.1/M.GRCh37.tsv.bgz
