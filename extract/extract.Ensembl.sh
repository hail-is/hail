#!/bin/bash

while getopts ":v:b:d:" args; do
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
    d)
      case $OPTARG in
        dna|cdna|cds|ncrna|pep)
          DATASET=$OPTARG
          ;;
        *)
          echo "Invalid dataset requested. -$OPTARG takes as an argument one of {dna,cdna,cds,ncrna,pep}". >&2
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
    URL="ftp://ftp.ensembl.org/pub/grch37/release-$VERSION/";
else
    URL="ftp://ftp.ensembl.org/pub/release-$VERSION/";
fi

if [ $DATASET == "dna" ]; then
  wget -c -O - "${URL}fasta/homo_sapiens/dna/Homo_sapiens.${BUILD}.dna.chromosome.*.fa.gz" | \
  zcat | \
  awk -v FS=$' ' -v OFS=$'\t' '
    BEGIN { 
      print "chromosome","position","reference_allele"; 
    }
    { 
      if ( $1 ~ /^>/ ) { 
        sequence=""; 
	split($3, locus, ":"); 
      } else { 
	sequence=sequence$0;
      } 
    } 
    END { 
      for ( i=locus[4]; i<=locus[5]; i++ ) { 
        allele=substr(sequence,i,1); 
    	if (allele != "N") { 
	  print locus[3],i,allele; 
	} 
      } 
    }' | \
  bgzip -c | \
  gsutil cp - gs://hail-datasets/raw-data/Ensembl/Ensembl_human_reference_dna_sequence.release_${VERSION}.${BUILD}.tsv.bgz
fi

