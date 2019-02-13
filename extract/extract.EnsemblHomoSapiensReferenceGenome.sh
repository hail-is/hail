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
    echo "${URL_ROOT}/fasta/homo_sapiens/dna/Homo_sapiens.${BUILD}.dna.chromosome.${i}.fa.gz";
  done ) |  
zcat |
awk -v FS=$' ' -v OFS=$'\t' '
  BEGIN { 
    print "chromosome","position","reference_allele"; 
  }
  { 
    if ( $1 ~ /^>/ ) {
      split($3, locus, ":");
      chromosome=locus[3];
      start=0;	
    } else {
      end=start+length($0); 
      for ( i=1; i<=length($0); i++ ) {
	allele=substr($0,i,1);
        if ( allele != "N" ) {
  	  print chromosome,start+i,allele;
	}
      }
      start=end;	
    } 
  }' | 
bgzip -c |
gsutil cp - gs://hail-datasets-extract/EnsemblHomoSapiensReferenceGenome_release-${RELEASE}_${BUILD}.tsv.bgz
 
