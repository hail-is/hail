#!/bin/bash

wget -c -O - "ftp://ftp.1000genomes.ebi.ac.uk/vol1/ftp/data_collections/1000G_2504_high_coverage/20130606_g1k_3202_samples_ped_population.txt" |
bgzip -c |
gsutil cp - gs://hail-datasets-tmp/1000_Genomes_NYGC_30x/1000_Genomes_NYGC_30x_samples_ped_population.txt.bgz
