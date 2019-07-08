#!/bin/bash

## Use flag "-r" to specify Ensembl release, e.g. "./extract.Ensembl_homo_sapiens_reference_genome.GRCh38.sh -r 96"

if [ $# -eq 0 ]
  then
    echo "Argument \"-r\" (Ensembl release version) must be specified."
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

wget -c -O - $( 
  for i in {1..22} {X,Y,MT}; do
    echo "${URL_ROOT}/fasta/homo_sapiens/dna/Homo_sapiens.GRCh38.dna.chromosome.${i}.fa.gz";
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
gsutil cp - gs://hail-datasets-raw-data/Ensembl/Ensembl_homo_sapiens_reference_genome_release${RELEASE}.GRCh38.tsv.bgz

