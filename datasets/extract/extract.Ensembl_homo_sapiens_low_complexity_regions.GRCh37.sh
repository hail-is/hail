#!/bin/bash

## Use flag "-r" to specify Ensembl release, e.g. "./extract.Ensembl_homo_sapiens_low_complexity_regions.GRCh37.sh -r 96"

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

URL_ROOT="ftp://ftp.ensembl.org/pub/grch37/release-${RELEASE}";

wget -c -O - $(
  for i in {1..22} {X,Y,MT}; do 
    echo "${URL_ROOT}/fasta/homo_sapiens/dna/Homo_sapiens.GRCh37.dna_sm.chromosome.${i}.fa.gz"; 
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
gsutil cp - gs://hail-datasets-raw-data/Ensembl/Ensembl_homo_sapiens_low_complexity_regions_release${RELEASE}.GRCh37.tsv.bgz

