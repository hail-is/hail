## <a name="methods"></a> Methods

### Summary Statistics

#### How do I generate basic summary statistics about my dataset such as number of samples and variants?

```
count
```


### Quality Control

#### How do I calculate QC metrics per variant?

```
variantqc
```

#### How do I calculate QC metrics per sample?

```
sampleqc
```


### Imputing Sex

#### How do I impute the sex of my samples?

```
imputesex -m 0.01 exportsamples -o /path/to/output.tsv -c "ID=s.id, Fstat=sa.imputesex.Fstat, ImputedSex=sa.imputesex.isFemale"
```


### Linear Regression

#### How do I perform linear regression on a subset of samples based on a sample annotation?

```
linreg -y ' if (sa.pheno.cohortName == "cohort1") sa.pheno.bloodPressure else NA: Double'
```

For optimal performance, make sure the Type defined after `NA:` is the same as the type of the response variable `sa.pheno.bloodPressure`.

### Variant Effect Predictor

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
