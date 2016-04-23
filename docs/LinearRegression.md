# Linear Regression

The `linreg` command computes, for each variant, the p-value of the t-test for the genotype coefficient of the linear function of best fit from sample genotype and covariates to
quantitative phenotype or case-control status.

Command line options:
 - `-y | --y <filename>` -- a sample annotation of numeric or Boolean type designating the response variable, i.e. phenotype
 - `-c | --covariates <filename>` -- a comma-separated list of sample annotations of numeric or Boolean type (optional)
 - `-r | --root <root>` -- variant annotation path for linreg output, starting with `va` (optional, default is `va.linreg`)

Assuming there are sample annotations `sa.pheno.height`, `sa.cov.age`, and `sa.cov.isMale`, the command
```
linreg -y sa.pheno.height -c sa.cov.age,sa.cov.isMale
```
fits a model of the form

`height = b0 + b1 * x + b2 * age + b3 * isMale + e`

where the genotype `x` is coded as 0 for HomRef, 1 for Het, and 2 for HomVar, the Boolean covariate isMale is coded as 1 for true (male) and 0 for false (female), and `e` is normal noise.

Five variant annotations are then added with root `va.linreg` as shown in the table. These annotations can then be accessed by other methods, including exporting to TSV with other variant annotations.

Annotation | Type | Value
---|---|---
v.contig | String | chromosome
v.pos | Int| position
v.ref | String | reference allele
v.alt | String | alternate allele
va.linreg.nMissing | Int | count of missing (imputed) genotypes
va.linreg.beta | Double | fit genotype coefficient, `b1` above
va.linreg.se | Double | standard error of beta
va.linreg.tstat | Double | t-statistic, equal to beta / se
va.linreg.pval | Double | p-value

Phenotype and covariate sample annotations may also be specified using [programmatic expressions](https://github.com/broadinstitute/hail/blob/master/docs/ProgrammaticAnnotation.md) without identifiers, such as `if (sa.isMale) sa.cov.age else (2 * sa.cov.age + 10)`.

The samples included in the regression are those in the variant data set with phenotype and all covariates defined. For each variant, missing genotypes are imputed as the mean of called genotypes. For Boolean types, true is coded as 1 and false as 0. In particular, for the sample annotation `sa.fam.isCase` added by importing a `.fam` file with case-control phenotype, case is 1 and control is 0.

The standard least-squares linear regression model is derived in Section 3.2 of [The Elements of Statistical Learning, 2nd Edition](https://web.stanford.edu/~hastie/local.ftp/Springer/OLD/ESLII_print4.pdf). See equation 3.12 for the t-statistic which follows the t-distribution with n - k - 2 degrees of freedom, under the null hypothesis of no effect, with n samples and k covariates in addition to genotype and intercept.