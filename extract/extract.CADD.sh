#!/bin/bash

while getopts ":v:b:" args; do
  case $args in
    v)
      VERSION=$OPTARG
      ;;
    b)
      case $OPTARG in
        GRCh37|GRCh38)
	  BUILD=$OPTARG
	  ;;
	*)
	  echo "Invalid argument: -$OPTARG takes either \"GRCh37\" or \"GRCh38\" as an argument." >&2
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

SNVS_URL="http://krishna.gs.washington.edu/download/CADD/v${VERSION}/${BUILD}/whole_genome_SNVs.tsv.gz"
INDELS_URL="http://krishna.gs.washington.edu/download/CADD/v${VERSION}/${BUILD}/InDels.tsv.gz"

wget -c -O - $SNVS_URL $INDELS_URL | \
zcat | \
grep -v '^#' | \
awk -v FS=$'\t' -v OFS=$'\t' 'BEGIN {print "chromosome","position","ref","alt","raw_score","PHRED_score"} {print $0}' | \
bgzip -c | \
gsutil cp - gs://hail-datasets-raw-data/CADD/CADD_v${VERSION}_${BUILD}.tsv.bgz

