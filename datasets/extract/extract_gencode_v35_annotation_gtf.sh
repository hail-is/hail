#!/bin/bash

GTF_URL="https://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_human/release_35/gencode.v35.annotation.gtf.gz"

wget -c -O - $GTF_URL | \
zcat | \
bgzip -c | \
gsutil -u broad-ctsa cp - gs://hail-datasets-tmp/GENCODE/gencode.v35.annotation.gtf.bgz
