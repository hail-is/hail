#!/usr/bin/env bash

# check that java is installed
# java -version (>1.7)

# when the analysis has started ?
date

#################################################################
###### SET ENVIRONMENT ##########################################
#################################################################

# path to the main directory of the course named WEScourse
# please change accordingly
SOURCE_DIR=`pwd`
# number of cpus used for the analysis
# please change accordingly
CORES=4

# GENOME DATA
REF_GENOME=$SOURCE_DIR/data/hg19/chr22.fa 
TOOLS=$SOURCE_DIR/tools
PICARD_TOOLS=$TOOLS/picard-tools-1.107
RAWDATA_DIR=$SOURCE_DIR/corpasome/fastq
GATK=$SOURCE_DIR/tools/gatk3
ANNOVAR=$TOOLS/annovar
TARGETS=$SOURCE_DIR/data/hg19/chr22.exons.bed
dbsnp=$SOURCE_DIR/data/hg19/dbsnp_138.b37.chr22.vcf
indels=$SOURCE_DIR/data/hg19/Mills_and_1000G_gold_standard.indels.b37.chr22.vcf
candidateGenes=$SOURCE_DIR/data/candidateGenes/schizophrenia.cand.genes.ids

# download GATK 1.8 version
cd $GATK
wget https://www.dropbox.com/s/2z70te3b1x8wcpa/GenomeAnalysisTK.1.8.jar
cd $SOURCE_DIR

#################################################################
### the corpas family 
#################################################################

father=dad
mother=mom
child=daughter

pedfile=$SOURCE_DIR/corpasome/pedigree/corpasome.ped

cd $SOURCE_DIR/corpasome

#################################################################
### prepare genome
#################################################################

### burrows-wheeler index of the ref genome
$TOOLS/bin/bwa index $REF_GENOME

### create fasta index file
$TOOLS/bin/samtools faidx $REF_GENOME

cd $GATK
wget 

####################################################################
### Read QC
### check: http://www.bioinformatics.babraham.ac.uk/projects/fastqc
###################################################################

mkdir fastqc
cd fastqc

for i in $father $mother $child
do
    echo "# fastqc $i processing "
    read1=$RAWDATA_DIR/${i}.chr22.1.fq
    read2=$RAWDATA_DIR/${i}.chr22.2.fq

    perl $TOOLS/FastQC/fastqc $read1 -o `pwd` >>$i.1.fastqc.log 2>>$i.1.fastqc.log

    perl $TOOLS/FastQC/fastqc $read2 -o `pwd` >>$i.2.fastqc.log 2>>$i.2.fastqc.log
    
    echo "# fastqc $i processing done."
done

#################################################################
### 1. Read mapping
#################################################################

### mapping using bwa
mkdir ../bam
cd ../bam

for i in $father $mother $child
do
    echo "# mapping $i reads started"
    read1=$RAWDATA_DIR/${i}.chr22.1.fq
    read2=$RAWDATA_DIR/${i}.chr22.2.fq
    readgroup='@RG\tID:'$i'\tPL:Illumina\tSM:corpasome_'$i'\tLB:unknown\tPU:unknown'        
  
    $TOOLS/bin/bwa mem -M -R $readgroup -t $CORES $REF_GENOME $read1 $read2 >$i.chr22.sam

    echo "# mapping $i reads done."
done

### sort bam file
for i in $father $mother $child
do
    echo "# sorting $i bam file started"
   
    $TOOLS/bin/samtools view -bS $i.chr22.sam | $TOOLS/bin/samtools sort - $i.chr22
   
    $TOOLS/bin/samtools index $i.chr22.bam
   
    echo "# sorting $i bam file done."
done

#################################################################
### 2. GATK3 best practices
### http://www.broadinstitute.org/gatk/guide/best-practices ->DNAseq
#################################################################

#######################
### 2.1 mark duplicates

### useful links 
### https://broadinstitute.github.io/picard/command-line-overview.html

for i in $father $mother $child
do
    echo "# mark duplicates for $i started"
   
    java -Xmx64g -jar -XX:-UsePerfData $PICARD_TOOLS/MarkDuplicates.jar \
        INPUT=$i.chr22.bam \
	    OUTPUT=$i.chr22.markdup.bam \
	    METRICS_FILE=$i.chr22.sorted.metrics.txt \
	    REMOVE_DUPLICATES=TRUE \
	    MAX_RECORDS_IN_RAM=80000000 

    java -jar -Xmx64g $PICARD_TOOLS/BuildBamIndex.jar I=$i.chr22.markdup.bam
    
    echo "# mark duplicates for $i done"
done

#########################
### 2.2 indel realignment

for i in $father $mother $child
do
    echo "# perform indel realignment for $i "
  
    input=$i.chr22.markdup.bam
    reference=$REF_GENOME
    output=$i.chr22.realn.bam
    intervals=$i.chr22.realn.intervals

    java -jar -Xmx64g -XX:-UsePerfData ${GATK}/GenomeAnalysisTK.1.8.jar -T RealignerTargetCreator -I $input -R $reference -o $intervals --fix_misencoded_quality_scores 
    
    java -jar -Xmx64g -XX:-UsePerfData ${GATK}/GenomeAnalysisTK.1.8.jar -T IndelRealigner --fix_misencoded_quality_scores \
                                                                    -targetIntervals ${intervals} -I ${input} -o ${output} -R ${reference} 
    
    echo "# indel realignment for $i done"
done

######################################
### 2.3 recalibrate basecalling scores

for i in $father $mother $child
do
  echo "# recalibration of basecalling scores started for $i .."
  
  input=$i.chr22.realn.bam
  reference=$REF_GENOME
  output=$i.chr22.recal.bam
  grp=$i.chr22.grp

  java -jar -XX:-UsePerfData $GATK/GenomeAnalysisTK.1.8.jar -T BaseRecalibrator -I ${input} -R ${reference} -o ${grp} -knownSites ${dbsnp} \
                                                        -knownSites $indels --filter_reads_with_N_cigar -rf BadCigar
  
  java -jar -XX:-UsePerfData $GATK/GenomeAnalysisTK.1.8.jar -T PrintReads -R ${reference} -I ${input} -BQSR ${grp} -o ${output} --filter_reads_with_N_cigar -rf BadCigar 
  
  java -jar $PICARD_TOOLS/BuildBamIndex.jar INPUT=$output 
  
  echo "# recalibration of basecalling scores done."
done

#################################################################
### 4. calculate coverage 
#################################################################

mkdir ../coverage
cd ../coverage

for i in $father $mother $child
do
    echo "# calculate exon $i coverages started..."
  
    $TOOLS/bin/coverageBed -abam ../bam/$i.chr22.recal.bam -b $TARGETS >$i.coverage

  
    java -Xmx2g -jar $GATK/GenomeAnalysisTK.1.8.jar -T DepthOfCoverage -R $REF_GENOME -o $i.depthOfCoverage.txt \
                                                -I ../bam/$i.chr22.recal.bam \
                                                -geneList:REFSEQ $ANNOVAR/humandb_chr22/hg19_refGene.txt \
                                                -L $TARGETS \
                                                -ct 1 -ct 5 -ct 10 -ct 20 -ct 50 
  
    echo "# calculate exon $i coverages done."

done

#################################################################
### 5. Variant calling
#################################################################

##################################################
### 5.1 call variants with GATK Haplotype Caller

### Unfortunately this step takes several hours, so we skip this for today
### but you can run it over night as an exercise writing your own script to execute the program 

#mkdir ../variants_gatk
#cd ../variants_gatk

#input1=../bam/$father.chr22.recal.bam 
#input2=../bam/$mother.chr22.recal.bam
#input3=../bam/$child.chr22.recal.bam
#output=corpasome.trio.chr22.htc.vcf

#java -Xmx64g -jar ${GATK}/GenomeAnalysisTK.1.8.ar -R $REF_GENOME \
#    		-T HaplotypeCaller \
#    		-I ${input1} \
#		-I ${input2} \
#		-I ${input3} \   
#		-L $TARGETS \
#    		--dbsnp ${dbsnp} \
#    		-stand_call_conf 30.0 \
#    		-stand_emit_conf 30.0 \
#		-minPruning 3 \
#		-nct $CORES \
#    		-o ${output} 2>haplotypecaller.log &

######################################################
### 5.2 call variants with samtools and bcftools

mkdir ../variants_samtools
cd ../variants_samtools

### generate mpileup file for all three samples together

$TOOLS/bin/samtools mpileup -u -l $TARGETS -f $REF_GENOME ../bam/$father.chr22.recal.bam ../bam/$mother.chr22.recal.bam ../bam/$child.chr22.recal.bam >corpasome.trio.chr22.raw.bcf 2>mpilup.log

### call variants with bcftools from bcffile

# generate bcf file
$TOOLS/bin/bcftools view -bvcg corpasome.trio.chr22.raw.bcf >corpasome.trio.chr22.snps.bcf
# filter bcffile by quality
$TOOLS/bin/bcftools view corpasome.trio.chr22.snps.bcf | $TOOLS/bin/vcfutils.pl varFilter -d 20  - >corpasome.trio.chr22.snps.filtered.vcf

#################################################################
### 6. Functional analysis
#################################################################

############################
### 6.1. preprocess vcf file

### break rows with multiple rows in the vcf file

$TOOLS/bin/vcfbreakmulti corpasome.trio.chr22.snps.filtered.vcf >corpasome.trio.chr22.snps.filtered.singleallele.vcf

### convert to a more simple readable vcf file format

$TOOLS/bin/vcf-to-testvariant corpasome.trio.chr22.snps.filtered.singleallele.vcf >corpasome.trio.chr22.snps.filtered.singleallele.testvariant.vcf

###########################################
### 6.2. annotate the variants with ANNOVAR

### prepare the annovar input file

input=corpasome.trio.chr22.snps.filtered.singleallele.testvariant.vcf
output=corpasome.trio.chr22.snps.filtered.annovarinput

perl $ANNOVAR/convert2annovar.pl --comment --includeinfo --format vcf4old $input | grep -v -P "^##" >$output

### run ANNOVAR

input=$output
output=corpasome.trio.chr22.snps.filtered

perl $ANNOVAR/table_annovar.pl $input $ANNOVAR/humandb_chr22 --outfile $output --remove --otherinfo --buildver hg19 \
                                -protocol refGene,1000g2012apr_eur,esp6500si_ea,snp138,ljb2_sift,ljb2_pp2hvar,clinvar_20140702 \
                                -operation g,f,f,f,f,f,f 2>annovar.log

addString=`grep "#CHROM" $output.hg19_multianno.txt | perl -ane '$_=~s/^#//;print $_;'`

perl $TOOLS/bin/changeHeader.pl $output.hg19_multianno.txt "^Chr" "Otherinfo$" " `echo $addString`" | grep -v -P "^#" >$output.annovar.tsv

#################################################################
### 6.3 Filter pipeline

### 6.3.1 filter the results for mode of inheritance: recessive 

input=$output.annovar.tsv
output=$output.annovar.recessive

perl $TOOLS/bin/filter/filterTestvariantForPattern.pl $input -i "corpasome_"$father",corpasome_"$mother",corpasome_"$child -p "01,01,11" >$output

### filter for MAF 

input=$output
output=$input.noncommon

perl $TOOLS/bin/filter/filterAnnovarBySmallerValue.pl $input -s esp6500si_ea,1000g2012apr_eur -v 0.05,0.05 >$output

### filter for potentially deleterious exonic variants

input=$output
output=$input.exonsplicing

perl $TOOLS/bin/filter/filterTestvariantForPatternOR.pl $input -i Func.refGene,ExonicFunc.refGene \
                                                        -p "^(splicing|exonic;splicing)$","(frameshift|nonsynonymous|stopgain|stoploss)" >$output

### filter for variants predicted to be deleterious by either SIFT or PolyPhen2

input=$output
output=$input.pred

perl $TOOLS/bin/filter/filterTestvariantForPatternOR.pl $input -i ljb2_sift,ljb2_pp2hvar -p "(D|P),(D|P)" >$output

### is the gene is our canditate gene list

input=$output
output=$input.cands

perl $TOOLS/bin/filter/filterColumnForList.pl $input -i Gene.refGene -l $candidateGenes -f 1 -e 1 >$output

### 6.2 filter for de novo variants

### denovogear (Ramu et al. 2013, Nature Methods)
### https://github.com/denovogear/denovogear

### run denovogear to detect denovo mutations

mkdir ../denovo
cd ../denovo

### run denovogear

bcffile=../variants_samtools/corpasome.trio.chr22.raw.bcf

$TOOLS/bin/denovogear dnm auto --ped $pedfile --bcf $bcffile >corpasome.trio.chr22.denovogear.txt 2>corpasome.denovogear.log

perl $TOOLS/bin/denovogear2tsv.pl corpasome.trio.chr22.denovogear.txt >corpasome.trio.chr22.denovogear.tsv

### annotate the denovo mutations using ANNOVAR

perl $TOOLS/bin/denovogear2annovar.pl corpasome.trio.chr22.denovogear.txt >corpasome.trio.chr22.denovogear.annovar_input

perl $ANNOVAR/table_annovar.pl corpasome.trio.chr22.denovogear.annovar_input $ANNOVAR/humandb_chr22 --otherinfo --buildver hg19 \
               -protocol refGene,1000g2012apr_eur,esp6500si_ea,snp138,ljb2_sift,ljb2_pp2hvar,clinvar_20140702 \
               -operation g,f,f,f,f,f,f 2>annovar.log

### output can be found in corpasome.trio.chr22.denovogear.annovar_input.hg19_multianno.txt
#less corpasome.trio.chr22.denovogear.annovar_input.hg19_multianno.txt

### filter down the results list for false positives

### filter by MAF: not found in other population databases

input=corpasome.trio.chr22.denovogear.annovar_input.hg19_multianno.txt
output=$input.noncommon

perl $TOOLS/bin/filter/filterAnnovarBySmallerValue.pl $input -s esp6500si_ea,1000g2012apr_eur -v 0.00001,0.00001 >$output

### filter out variants already found in dbSNP

input=$output
output=$input.notdbSNP

perl $TOOLS/bin/filter/filterAnnovarByIsEmpty.pl $input -s snp138 >$output

### only potentially deleterious exonic variants

input=$output
output=$input.exonsplicing

perl $TOOLS/bin/filter/filterTestvariantForPatternOR.pl $input -i Func.refGene,ExonicFunc.refGene \
         -p "^(splicing|exonic;splicing)$","(frameshift|nonsynonymous|stopgain|stoploss)" >$output

### predicted to be deleterious by either SIFT or PolyPhen2

input=$output
output=$input.pred

perl $TOOLS/bin/filter/filterTestvariantForPatternOR.pl $input -i ljb2_sift,ljb2_pp2hvar -p D,D >$output

### is the gene is our canditate gene list

input=$output
output=$input.cands

perl $TOOLS/bin/filter/filterColumnForList.pl $input -i Gene.refGene -l $candidateGenes -f 1 -e 1 >$output

###########################################################
# when the analysis has ended ??
date

