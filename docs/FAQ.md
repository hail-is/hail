# Frequently Asked Questions

Skip to:
 - [Acronym Definitions](#acronyms)
 - [Data Import](#dataimport)
 - [Methods](#methods)
 - [Working with Annotations](#annotations)
 - [Filtering Data](#filtering)
 - [Variant Effect Predictor](#vep)
 - [Data Export](#dataexport)
 - [Pipeline Optimization](#pipeopt)
 

<a name="acronyms"></a>
## Acronym Definitions

Acronym | Long Form | Description | Resource
--- | :-: | ---
VDS | Variant Dataset | Hail's format for storing data | dsfasfda
VCF | Variant Call Format | A text file representation for storing genetic data | adfsads
VEP | Variant Effect Predictor | A tool for annotating exomes | adfsasdfa
QC | Quality Control | 


<a name="dataimport"></a>
## Data Import

#### How do I import data from a VCF file?

#### How are insertions or deletion variants coded in the VDS?

<a name="methods"></a>
## Methods

#### How do I generate basic summary statistics about my dataset such as number of samples and variants?

```
count
```

#### How do I calculate QC metrics per variant?

```
variantqc
```

#### How do I calculate QC metrics per sample?

```
sampleqc
```

#### How do I impute the sex of my samples?

```
imputesex -m 0.01 exportsamples -o /path/to/output.tsv -c "ID=s.id, Fstat=sa.imputesex.Fstat, ImputedSex=sa.imputesex.isFemale"
```


#### Does Hail handle sex chromosomes differently in variantqc and sampleqc?

#### How do I perform linear regression on a subset of samples based on a sample annotation?

```
linreg -y ' if (sa.pheno.cohortName == "cohort1") sa.pheno.bloodPressure else NA: Double'
```

For optimal performance, make sure the Type defined after `NA:` is the same as the type of the response variable `sa.pheno.bloodPressure`.

<a name="annotations"></a>
## Working with Annotations

#### How can I find the current schema of variant, sample, and global annotations?

```
printschema
```

#### How do I count the number of samples matching a phenotype annotation?

```
annotateglobal expr -c '
    global.nMales = samples.count(sa.pheno.sex == "Male"),
    global.nFemales = samples.count(sa.pheno.sex == "Female"),
    global.nSamples = samples.count(true)'
```

#### Do I need to define the types when using `annotatesamples table`?

Hail does not currently infer the type based on input. The default type is `String`. See the documentation here.

#### How do I import annotations from a PLINK fam file?

```
annotatesamples fam -i myStudy.fam -q
```

Use the `-q` flag for a quantitative phenotype.

#### How do I make a new annotation that combines annotations currently in the dataset?

```
annotatevariants expr -c 'va.combAnnot = va.annot1 + ":" + va.annot2'
```

#### How does Hail annotate variants overlapping different intervals in an interval list?

#### How do I input phenotype information into Hail?

#### How do I create an annotation for only a subset of samples based on an existing annotation?

```
annotatesamples expr -c 'if (sa.pheno.cohortName == "cohort1") sa.pheno.bloodPressure else NA: Double'
```

For optimal performance, make sure the Type defined after `NA:` is the same as the type of the response variable `sa.pheno.bloodPressure`.

<a name="filtering"></a>
## Filtering Data

#### How do I create a sites-only VCF?

```
filtersamples all exportvcf -o /path/to/sitesonly.vcf.bgz
```

#### How do I filter out samples from a text file (one sample ID per line)?

```
filtersamples list --remove -i /path/to/badSamples.sample_list
```

#### Should I write a VDS every time I do a filter?

#### How do I filter genotypes based on meta-information in the VCF?
 
``` 
filtergenotypes --remove -c 'g.dp < 10 || 
   (g.ad[0] + g.ad[1]) / g.dp < 0.9 || 
   (g.isHomRef && (g.ad[0] / g.dp < 0.9 || g.gq < 25)) ||
   (g.isHet && (g.ad[1] / g.dp < 0.25 || g.pl[0] < 25)) ||
   (g.isHomVar && (g.ad[1] / g.dp < 0.9 || g.pl[0] < 25))'
```

#### How do I subset the dataset to only include non-psuedoautosomal variants on the X chromosome?

```
filtervariants expr --keep -c 'v.contig == "X" && !v.inParX'
```

#### How do I remove samples from my dataset based on summary statistics calculated from sample QC?

```
sampleqc
filtersamples expr --keep -c 'sa.qc.callRate >= 0.95'
```

#### How do I remove variants from my dataset based on summary statistics calculated from variant QC?

```
variantqc
filtervariants expr --keep -c 'va.qc.callRate >= 0.95 && va.qc.dpMean >= 20 && va.qc.pHWE > 1e-6 && va.pass'
```

<a name="vep"></a>
## VEP

#### How do I annotate my data using VEP?
 
```
hail importvcf /path/to/mydata.vcf.bgz \
splitmulti \
vep --config /path/to/vep.properties \
write -o /path/to/mydata.vep.vds
```


#### How do I only VEP annotate coding regions?
  
```
drop genotypes
filter to sites you want
vep this sites only vds
annotate original with vepped sites only
```

#### How do I parse the variant annotations from VEP to find the worst functional consequence?

#### How do I find all variants where the functional change on the canonical transcript results in a missense mutation?

<a name="dataexport"></a>
## Data Export

#### When exporting the VDS to a VCF file, what file extension should I use?

.vcf.bgz

#### How do I create a sites-only VCF?

```
filtersamples all exportvcf -o /path/to/sitesonly.vcf.bgz
```

#### How can I export all global annotations to a file?

#### How do I export my data so there are separate VCFs per chromosome?

#### How do I substitute a custom string for the rsID field in the VCF exported by Hail?
 
```
annotatevariants expr -c 'va.rsid = va.contig + "_" + va.start + "_" + va.ref + "_" + va.alt'
exportvcf /path/to/myvcf.vcf.bgz
```

<a name="pipeopt"></a>
## Pipeline Optimization

#### When should I write my data to a VDS file?

Whenever an execution happens, it needs a source.  If the first thing in that chain is a read, it’s reading a VDS (fast).  If the first thing is a vcf, it’s parsing the vcf every time (very slow)
Write to a VDS first.  It will be orders of magnitude faster.
yeah, just do an import/write/read at the beginning

