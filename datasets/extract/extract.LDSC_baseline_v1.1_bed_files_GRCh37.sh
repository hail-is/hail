#!/bin/bash

wget -O /tmp/baseline_v1.1_bedfiles.tgz https://data.broadinstitute.org/alkesgroup/LDSCORE/baseline_v1.1_bedfiles.tgz
tar -C /tmp/ -xzvf /tmp/baseline_v1.1_bedfiles.tgz
for file in /tmp/baseline_v1.1/*.bed; do
    echo $file;
    bgzip -c $file | gsutil cp - gs://hail-datasets-raw-data/LDSC/baseline_v1.1/bed_files/$(basename ${file}).bgz;
done
