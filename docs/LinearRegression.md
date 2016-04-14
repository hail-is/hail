# Linear Regression

The `linreg` command computes, for each variant, the p-value of the t-test for the genotype coefficient of the linear function of best fit from sample genotype and covariates to
quantitative phenotype or case-control status.

Command line options:
 - `-y | --y <filename>` -- a sample annotation of numeric or Boolean type designating the response variable, i.e. phenotype
 - `-c | --covariates <filename>` -- a comma-separated list of sample annotations of numeric or Boolean type (optional)
 - `-r | --root <root>` -- variant annotation path for linreg output, starting with `va` (optional, default is `va.linreg`)
 - `-o | --output <filename>` --  filename for default output TSV file (optional)

Assuming there are sample annotations `sa.pheno.height`, `sa.cov.age`, and `sa.cov.isMale`, the command
```
linreg -y sa.pheno.height -c sa.cov.age,sa.cov.isMale
```
adds the five variant annotations shown below with root `va.linreg`. Adding `-o myStudy.linreg` will additionally save a TSV file `myStudy.linreg` with a row for each variant and the following columns.

Annotation | Column Name | Value
---|---|---
v.contig | Chrom | chromosome
v.pos | Pos | position
v.ref | Ref | reference allele
v.alt | Alt | alternate allele
va.linreg.nMissing | Missing | count of missing genotypes
va.linreg.beta | Beta | genotype coefficient
va.linreg.se | StdErr | standard error
va.linreg.tstat | TStat | t-statistic
va.linreg.pval | PVal | p-value

Phenotype and covariate sample annotations may also be specified using [programmatic expressions](https://github.com/broadinstitute/hail/blob/master/docs/ProgrammaticAnnotation.md) without identifiers, such as `if (sa.isMale) sa.cov.age else (2 * sa.cov.age + 10)`.

The samples included in the regression are those in the variant data set with phenotype and all covariates defined. For each variant, missing genotypes are imputed as the mean of called genotypes. Cases are coded as 1 and controls are coded as 0. For Boolean types, true is coded as 1 and false as 0.

The standard least-squares linear regression model is derived in Section 3.2 of [The Elements of Statistical Learning, 2nd Edition](https://web.stanford.edu/~hastie/local.ftp/Springer/OLD/ESLII_print4.pdf). See equation 3.12 for the t-statistic which follows the t-distribution with n - k - 2 degrees of freedom, under the null hypothesis of no effect, with n samples and k covariates in addition to genotype and intercept.