# Group-based tests in Hail
  
## SKAT-O
 
SKAT-O is not implemented natively in Hail. Instead, we use the SKAT package in R to run SKAT-O as described [here](https://cran.r-project.org/web/packages/SKAT/vignettes/SKAT.pdf).

### Requirements
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

### Implementation Details

**All Hail variants and samples in the variant dataset at the time of calling `grouptest skato` are included in the SKAT-O test** 

1. Hail groups variants together by the keys specified by the `-k | --group-keys` command line option.
2. Duplicate variants are removed (one is randomly chosen). // FIXME: To-do item
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


### Running SKATO in Hail

**Example Command:**
```

```

**Input Specification:**

Flag | Description | Required | Default
:-: | ---
`-k | --group-keys` | Comma-separated list of annotations to be used as grouping variable(s) (must be attribute of `va`) | True | |
`-q | --quantitative` | y is a quantitative phenotype | False | False
`-y` | Response sample annotation (must be attribute of `sa`) | True | |
`-c | --covariates` | Covariate sample annotations, comma-separated (must be attribute of `sa`) | False | |
`-o | --output` | Path of output .tsv file | True | |

**Configuration Options:**

Flag | Description | Required | Default
:-: | ---
`--config` | Configuration file | True | |
`--block-size` | # of groups tested per invocation | False | 1000
`--seed` | Number to set seed to in R | False | 1
`--random-seed` | Use a random seed. Overrides `--seed` if set to True | False | False

**SKAT Null Model Options:**

Flag | Description | Required | Default
:-: | ---
`--n-resampling` | Number of times to resample residuals | False | 0
`--type-resampling` | Resampling method. One of [bootstrap, bootstrap.fast] | False | bootstrap
`--no-adjustment` | No adjustment for small sample sizes (applicable to non-quantitative phenotypes only) | False | False

**SKAT Options:**

Flag | Description | Required | Default
:-: | ---
`--kernel` | SKAT-O kernel type. One of [linear, linear.weighted, IBS, IBS.weighted, quadratic, 2wayIX] | False | linear.weighted
`--method` | Method for calculating p-values. One of [davies, liu, liu.mod, optimal.adj, optimal] | False | davies
`--weights-beta` | Comma-separated parameters for beta function for calculating weights. | False | 1,25
`--impute-method` | Method for imputing missing genotypes. One of [fixed, random, bestguess] | False | fixed
`--r-corr` | rho parameter for the unified test. rho=0 is SKAT only. rho=1 is Burden only. Can also be multiple comma-separated values | False | 0.0
`--missing-cutoff` | Missing rate cutoff for variant inclusion | False | 0.15
`--estimate-maf` | Method for estimating MAF. 1 = use all samples to estimate MAF. 2 = use samples with non-missing phenotypes and covariates | False | 1


### Output File

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