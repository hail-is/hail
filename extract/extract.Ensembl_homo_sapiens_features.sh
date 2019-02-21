#!/bin/bash

while getopts ":r:b:" args; do
  case $args in
    r)
      RELEASE=$OPTARG
      ;;
    b)
      case $OPTARG in
        GRCh37|GRCh38)
          BUILD=$OPTARG
          ;;
        *)
          echo "Invalid reference genome build: -$OPTARG takes either \"GRCh37\" or \"GRCh38\" as an argument." >&2
          exit 1
          ;;
      esac
      ;;
    \?)
      echo "Invalid option: -$OPTARG" >&2
      exit 1
      ;;
    :)
      echo "Option -$OPTARG requires an argument." >&2
      exit 1
      ;;
  esac
done

if [ $BUILD == "GRCh37" ]; then
    if (( $RELEASE > 87 )); then
        echo "Invalid release: no releases after 87 available for reference genome build \"GRCh37\"." >&2
        exit 1;
    fi
    URL_ROOT="ftp://ftp.ensembl.org/pub/grch37/release-${RELEASE}";
else
    URL_ROOT="ftp://ftp.ensembl.org/pub/release-${RELEASE}";
fi

wget -c -O - "${URL_ROOT}/gff3/homo_sapiens/Homo_sapiens.${BUILD}.${RELEASE}.gff3.gz" | 
zcat |
bgzip -c |
gsutil cp - gs://hail-datasets-raw-data/Ensembl/Ensembl_homo_sapiens_features_release${RELEASE}_${BUILD}.gff3.bgz

