# Linear Regression

The `linreg` command computes, for each variant, the linear function of best fit from sample genotype and covariates to
case-control status, outputing the p-value of the t-test for the genotype coefficient.

Command line options:
 - `-f | --fam <filename>` -- a [Plink .fam file](https://www.cog-genomics.org/plink2/formats#fam)
 - `-c | --cov <filename>` -- a .cov file, see below
 - `-o | --output <filename>` -- output TSV file

The command
```
linreg -f myStudy.fam -c myStudy.cov -o myStudy.linreg
```
outputs a TSV file `myStudy.linreg` with a row for each variant and the following columns.

Column | Value
---|---
CHR | chromosome
POS | position
REF | reference allele
ALT | alternate allele
MISS | count of missing genotypes
BETA | genotype coefficient
SE | standard error
T | t-statistic
P | p-value

A `.cov` file is a TSV file of sample covariate data. The first column records the sample ID. Here is an example with two samples:

```
IID   AGE   SEX   PC1   PC2
Ann   10    2     1.2   6.7
Bob   12    1     -0.2  2.8
```


Samples are included in the regression if and only if they are in the variant data set, the .cov file, and the .fam file with a defined phenotype. For each variant, missing genotypes are imputed as the mean of called genotypes.

The standard least-squares linear regression model is derived in Section 3.2 of [The Elements of Statistical Learning, 2nd Edition](https://web.stanford.edu/~hastie/local.ftp/Springer/OLD/ESLII_print4.pdf). See equation 3.12 for the t-statistic which follows the t-distribution with n - k - 2 degrees of freedom, under the null hypothesis of no effect, with n samples and k covariates in addition to genotype.