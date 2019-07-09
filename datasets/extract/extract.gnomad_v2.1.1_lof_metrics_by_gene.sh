#!/bin/bash
set -ex

wget -c -O - https://storage.googleapis.com/gnomad-public/release/2.1.1/constraint/gnomad.v2.1.1.lof_metrics.by_gene.txt.bgz | \
zcat | \
gzip -c > tofile


