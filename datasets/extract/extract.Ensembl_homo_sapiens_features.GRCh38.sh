#!/bin/bash


## Use flag "-r" to specify Ensembl release, e.g. "./extract.Ensembl_homo_sapiens_features.GRCh38.sh -r 96"

if [ $# -eq 0 ]
  then
    echo "Argument \"-r\" (Ensembl release version)  must be specified."
    exit 1
fi

while getopts ":r:" args; do
  case $args in
    r)
      RELEASE=$OPTARG
      ;;
    \?)
      echo "Invalid option: -$OPTARG" >&2
      exit 1
      ;;
  esac
done

URL_ROOT="ftp://ftp.ensembl.org/pub/release-${RELEASE}";

wget -c -O - "${URL_ROOT}/gff3/homo_sapiens/Homo_sapiens.GRCh38.${RELEASE}.gff3.gz" | 
zcat |
bgzip -c |
gsutil cp - gs://hail-datasets-raw-data/Ensembl/Ensembl_homo_sapiens_features_release${RELEASE}.GRCh38.gff3.bgz

