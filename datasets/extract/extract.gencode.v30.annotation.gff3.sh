#!/bin/bash
set -ex

wget -c -O - ftp://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_human/release_30/gencode.v30.annotation.gff3.gz | \
zcat | \
gzip -c > tofile
