#!/bin/bash


## Use flag "-r" to specify Ensembl release, e.g. "./extract.Ensembl_homo_sapiens_features.GRCh37.sh -r 87"

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

if (( $RELEASE > 87 )); then
  echo "Invalid release: no releases after 87 available for reference genome build \"GRCh37\"." >&2
  exit 1;
fi
URL_ROOT="ftp://ftp.ensembl.org/pub/grch37/release-${RELEASE}";

wget -c -O - "${URL_ROOT}/gff3/homo_sapiens/Homo_sapiens.GRCh37.${RELEASE}.gff3.gz" | 
zcat |
bgzip -c |
gsutil cp - gs://hail-datasets-raw-data/Ensembl/Ensembl_homo_sapiens_features_release${RELEASE}.GRCh37.gff3.bgz

