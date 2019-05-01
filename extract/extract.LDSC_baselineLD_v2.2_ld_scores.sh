#!/bin/bash

wget -O /tmp/1000G_Phase3_baselineLD_v2.2_ldscores.tgz https://data.broadinstitute.org/alkesgroup/LDSCORE/1000G_Phase3_baselineLD_v2.2_ldscores.tgz
mkdir /tmp/baselineLD_v2.2
tar -C /tmp/baselineLD_v2.2 -xzvf /tmp/1000G_Phase3_baselineLD_v2.2_ldscores.tgz

zcat /tmp/baselineLD_v2.2/baselineLD.1.l2.ldscore.gz > /tmp/baselineLD_v2.2/baselineLD_v2.2_ld_scores.tsv
for i in {2..22}; do
    zcat /tmp/baselineLD_v2.2/baselineLD.${i}.l2.ldscore.gz | tail -n+2 >> /tmp/baselineLD_v2.2/baselineLD_v2.2_ld_scores.tsv;
done
bgzip -c /tmp/baselineLD_v2.2/baselineLD_v2.2_ld_scores.tsv | gsutil cp - gs://hail-datasets-raw-data/LDSC/baselineLD_v2.2/baselineLD_v2.2_ld_scores.tsv.bgz

zcat /tmp/baselineLD_v2.2/baselineLD.1.annot.gz > /tmp/baselineLD_v2.2/baselineLD_v2.2_annotations.tsv
for i in {2..22}; do
    zcat /tmp/baselineLD_v2.2/baselineLD.${i}.annot.gz | tail -n+2 >> /tmp/baselineLD_v2.2/baselineLD_v2.2_annotations.tsv;
done
bgzip -c /tmp/baselineLD_v2.2/baselineLD_v2.2_annotations.tsv | gsutil cp - gs://hail-datasets-raw-data/LDSC/baselineLD_v2.2/baselineLD_v2.2_annotations.tsv.bgz

zcat /tmp/baselineLD_v2.2/baselineLD.1.l2.ldscore.gz | head -1 | cut -f4- | tr '\t' '\n' | awk 'BEGIN {print "annotation"} {print}' > /tmp/baselineLD_v2.2/header.txt
awk '{for (i=1; i<=NF; i++) { sums[i]+=$i; } } END {print "M_5_50"; for (c in sums) { print sums[c]; }}' /tmp/baselineLD_v2.2/baselineLD.*.l2.M_5_50 > /tmp/baselineLD_v2.2/M_5_50_counts.txt
paste /tmp/baseline_v1.1/header.txt /tmp/baselineLD_v2.2/M_5_50_counts.txt | bgzip -c | gsutil cp - gs://hail-datasets-raw-data/LDSC/baselineLD_v2.2/baselineLD_v2.2_M_5_50.tsv.bgz

awk '{for (i=1; i<=NF; i++) { sums[i]+=$i; } } END {print "M"; for (c in sums) { print sums[c]; }}' /tmp/baselineLD_v2.2/baselineLD.*.l2.M > /tmp/baselineLD_v2.2/M_counts.txt
paste /tmp/baselineLD_v2.2/header.txt /tmp/baselineLD_v2.2/M_counts.txt | bgzip -c | gsutil cp - gs://hail-datasets-raw-data/LDSC/baselineLD_v2.2/baselineLD_v2.2_M.tsv.bgz
