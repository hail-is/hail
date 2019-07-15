#!/bin/bash

URL_ROOT="http://ftp.1000genomes.ebi.ac.uk/vol1/ftp/release/20130502";

wget -c -O - "${URL_ROOT}/integrated_call_samples_v3.20130502.ALL.panel" |
bgzip -c |
gsutil cp - gs://hail-datasets-raw-data/1000_Genomes/1000_Genomes_phase3_samples.tsv.bgz

wget -c -O - "${URL_ROOT}/integrated_call_samples_v2.20130502.ALL.ped" |
bgzip -c |
gsutil cp - gs://hail-datasets-raw-data/1000_Genomes/1000_Genomes_phase3_sample_relationships.tsv.bgz

