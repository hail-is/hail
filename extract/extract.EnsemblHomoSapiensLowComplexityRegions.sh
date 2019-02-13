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
    URL_ROOT="ftp://ftp.ensembl.org/pub/grch37/release-${RELEASE}";
else
    URL_ROOT="ftp://ftp.ensembl.org/pub/release-${RELEASE}";
fi

wget -c -O - $(
  for i in {1..22} {X,Y,MT}; do 
    echo "${URL_ROOT}/fasta/homo_sapiens/dna/Homo_sapiens.${BUILD}.dna_sm.chromosome.${i}.fa.gz"; 
  done ) | 
zcat | 
awk -v FS=$' ' -v OFS=$'\t' '
  BEGIN {
    print "chromosome","start","end";
  }
  {
    if ( $1 ~ /^>/ ) {
      split($3, locus, ":");
      chromosome=locus[3];
      line_start=0;
      start=0;
      end=0;
      on=0;
    } else {
      for ( i=1; i<=length($0); i++ ) {
        allele=substr($0,i,1);
        if ( allele ~ /[atcg]/ ) {
          if (on) {
            end+=1;
          } else {
            start=line_start+i;
            end=line_start+i+1;
            on=1;
          }
        } else {
          if (on) {
            print chromosome,start,end;
            on=0;
          }
        }
      }
      line_start+=length($0);
    }
  }
  END {
    if (on) {
      print chromosome,start,end;
    }
  }' | 
bgzip -c |
gsutil cp - gs://hail-datasets-extract/EnsemblHomoSapiensLowComplexityRegions_release-${RELEASE}_${BUILD}.tsv.bgz

