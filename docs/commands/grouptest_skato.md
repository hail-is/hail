<div class="cmdhead"></div>

<div class="description"></div>

<div class="synopsis"></div>

<div class="options"></div>
 
<div class="cmdsubsection">

### Requirements:

SKAT-O is not implemented natively in Hail. Instead, we use the SKAT package in R to run SKAT-O as described [here](https://cran.r-project.org/web/packages/SKAT/vignettes/SKAT.pdf).

1. [R installed ( > 2.13.0 )](https://www.r-project.org)
2. The following R libraries installed:
    * [jsonlite](https://cran.r-project.org/web/packages/jsonlite/index.html)
    * [SKAT](https://cran.r-project.org/web/packages/SKAT/index.html)
3. A configuration text file pointing to the path of the Rscript binary (`hail.skato.Rscript`) and the path of the Hail R script to run SKAT (`hail.skato.script`). Use the `--config` command line option to specify the path to this file.
 
**Example configuration file:**
 
```
hail.skato.Rscript /usr/bin/Rscript
hail.skato.script /path/to/hail/src/dist/scripts/skato.r
```
</div>

<div class="cmdsubsection">

### Output File:

Results are written to the output file specified by `-o` or `--output` and have the following format:

Column Number | Name | Description
:-: | ---
1 | groupName | Name of group. Multiple group keys specified by `-k | --group-keys` are comma-separated
2 | pValue | P value from SKAT-O. Includes small-sample size adjustment if N < 2000 and non-quantitative phenotype
3 | pValueNoAdj | P value without adjustment for small-sample size. `null` if N \ge 2000 or quantitative phenotype
4 | nMarker | Number of markers in group before filtering for missing rate
5 | nMarkerTest | Number of markers tested in group

**Example:**

```
groupName       pValue  pValueNoAdj     nMarker nMarkerTest
group_4 0.2932  0.2447  8       2
group_0 0.6367  0.6292  8       2
group_10        0.06804 0.08229 7       3
group_3 0.7344  0.7679  8       1
group_5 0.607   0.582   8       2
```

**Notes:**

- If the number of markers tested is 0, SKAT-O returns a P-value of 1.0
- Results are not sorted by groupName
- An empty set or array as the group key will produce an empty string as the groupName

</div>

<div class="cmdsubsection">
### Implementation Details:

**All Hail variants and samples in the variant dataset at the time of calling `grouptest skato` are included in the SKAT-O test** 

1. Hail groups variants together by the keys specified by the `-k | --group-keys` command line option.
2. If there are duplicate variants, one is randomly chosen.
3. The phenotype vector and covariate matrix are constructed from the command-line options `-y` and `-c | --covariates` respectively. 
4. Genotype vectors are constructed using the value in `g.nNonRefAlleles`
5. The following R commands are run:

```
# 1. Run Null Model
## No covariates specified
obj <- SKAT_Null_Model(Y ~ 1, ... )

## Covariates specified (X)
obj <- SKAT_Null_Model(Y ~ X, ... )

# 2. Run SKAT once per group with genotype matrix (Z) and result of null model
SKAT(Z, obj, ... )

# Y is the phenotype vector (coded as 0/1 for dichotomous variable and NA for missing value)
# ... are optional parameters such as impute.method, r.corr, etc.
```

**Caveats**

Hail will not return the same answer as SKAT-O from PLINK files:

 * when variants have a minor allele frequency (MAF) of 0.5. This is because SKAT-O cannot know which allele is the reference allele from .bim files. Therefore, in Hail, the genotype vector codes reference allele homozygotes as 0 while in SKAT-O from PLINK files, the genotype vector codes these as 2.
 * when using the `--estimate-maf 2` option if variants have a MAF equal to 0.5 after filtering samples with missing phenotypes and/or covariates. (see above)
 * when using the `--impute-method random` option.

</div>

<div class="cmdsubsection">

### Examples:

<h4 class="example">Gene burden test from VEP annotation with SKAT-O</h4>
```
hail importvcf /path/to/file.vcf \
variantqc \
vep --config /path/to/vep.properties \
filtervariants expr --keep -c 'va.qc.AF < 0.01 && va.qc.AC > 0' \
annotatesamples table -i /path/to/sampleAnnotations.txt -r sa.pheno \
grouptest skato 
    -k "let x = va.vep.transcript_consequences.map(csq => csq.gene_id).toSet in if (x.size == 0) NA: Set[String] else x" 
    -y "sa.pheno.Phenotype" 
    -c "sa.pheno.PC1,sa.pheno.PC2" -o /path/to/skatoOutput.tsv 
```
</div>