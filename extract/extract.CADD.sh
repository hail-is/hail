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

if !$VERSION; then
  echo "-v argument must be supplied to indicate dataset version." >&2
  exit 1
fi

if !$BUILD; then
  echo "-b argument must be supplied to indicate reference genome build." >&2
  exit 1
fi

SNVS_URL="http://krishna.gs.washington.edu/download/CADD/${VERSION}/${BUILD}/whole_genome_SNVs.tsv.gz"
INDELS_URL="http://krishna.gs.washington.edu/download/CADD/${VERSION}/${BUILD}/InDels.tsv.gz"

wget -c -O - $SNVS_URL $INDELS_URL | \
zcat | \
grep -v '^#' | \
awk -v FS=$'\t' -v OFS=$'\t' 'BEGIN {print "chromosome","position","ref","alt","raw_score","PHRED_score"} {print $0}' | \
bgzip -c | \
gsutil cp - gs://hail-datasets/raw-data/CADD/CADD.${VERSION}.${BUILD}.tsv.bgz

