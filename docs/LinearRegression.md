# Linear regression in Hail

The `linreg` module computes, for each variant, the linear function of best fit from sample genotype and covariates to
case-control status, outputing the p-value of the t-test for the genotype coefficient.

Command line arguments:
 - `-f | --fam <filename>` -- a [Plink .fam file](https://www.cog-genomics.org/plink2/formats#fam)
 - `-c | --cov <filename>` -- a .cov file, see below
 - `-o | --output <fileroot>` -- a root name for output files

The command
```
linreg -f myStudy.fam -c myStudy.cov -o myStudy
```
outputs a hadoop tsv folder `myStudy.linreg` with header:

`CHR POS REF ALT MISS BETA SE T P`

The last five columns record number of missing genotypes, genotype coefficient, standard error, t-statistic, and p-value, respectively.

A `.cov` file is a tsv file of sample covariate data. The first column records the sample ID. Here is an example with two samples:

```
IID   AGE   SEX   PC1   PC2
Ann   10    2     1.2   6.7
Bob   12    1     -0.2  2.8
```


Samples are included in the regression if and only if they are in the variant data set, the .cov file, and the .fam file with a defined phenotype. For each variant, missing genotypes are imputed as the mean of called genotypes.

The linear regression model is derived in Section 3.2 of [The Elements of Statistical Learning, 2nd Edition](https://web.stanford.edu/~hastie/local.ftp/Springer/OLD/ESLII_print4.pdf). See equation 3.12 for the t-statistic.