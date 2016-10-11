<div class="cmdhead"></div>

<div class="description"></div>

<div class="synopsis"></div>

<div class="options"></div>

<div class="cmdsubsection">
### Implementation Details

The `linreg` command computes, for each variant, statistics of the $t$-test for the genotype coefficient of the linear function of best fit from sample genotype and covariates to quantitative phenotype or case-control status. Hail only includes samples for which phenotype and all covariates are defined. For each variant, Hail imputes missing genotypes as the mean of called genotypes.

Assuming there are sample annotations `sa.pheno.height`, `sa.cov.age`, `sa.cov.isFemale`, and `sa.cov.PC1`, the command
```
linreg -y sa.pheno.height -c sa.cov.age,sa.cov.isFemale,sa.cov.PC1
```
considers a model of the form
$$
\mathrm{height} = \beta_0 + \beta_1 \, \mathrm{gt} + \beta_2 \, \mathrm{age} + \beta_3 \, \mathrm{isFemale} + \beta_4 \, \mathrm{PC1} + \varepsilon, \quad \varepsilon \sim \mathrm{N}(0, \sigma^2)
$$
where the genotype $\mathrm{gt}$ is coded as $0$ for HomRef, $1$ for Het, and $2$ for HomVar, and the Boolean covariate $\mathrm{isFemale}$ is coded as $1$ for true (female) and $0$ for false (male). The null model sets $\beta_1 = 0$.

Four variant annotations are then added with root `va.linreg` as shown in the table. These annotations can then be accessed by other methods, including exporting to TSV with other variant annotations.

Annotation | Type | Value
---|---|---
`va.linreg.beta` | Double | fit genotype coefficient, $\hat\beta_1$
`va.linreg.se` | Double | estimated standard error, $\widehat{\mathrm{se}}$
`va.linreg.tstat` | Double | $t$-statistic, equal to $\hat\beta_1 / \widehat{\mathrm{se}}$
`va.linreg.pval` | Double | $p$-value

Phenotype and covariate sample annotations may also be specified using [programmatic expressions](reference.html#HailExpressionLanguage) without identifiers, such as
```
if (sa.isMale) sa.cov.age else (2 * sa.cov.age + 10)
```
For Boolean types, true is coded as $1$ and false as $0$. In particular, for the sample annotation `sa.fam.isCase` added by importing a `.fam` file with case-control phenotype, case is $1$ and control is $0$.

Hail's linear regression test corresponds to the `q.lm` test in [EPACTS](http://genome.sph.umich.edu/wiki/EPACTS#Single_Variant_Tests). For each variant, Hail imputes missing genotypes as the mean of called genotypes, whereas EPACTS subsets to those samples with called genotypes. Hence, Hail and EPACTS results will currently only agree for variants with no missing genotypes.

The standard least-squares linear regression model is derived in Section 3.2 of [The Elements of Statistical Learning, 2nd Edition](https://web.stanford.edu/~hastie/local.ftp/Springer/OLD/ESLII_print4.pdf). See equation 3.12 for the t-statistic which follows the t-distribution with $n - k - 2$ degrees of freedom, under the null hypothesis of no effect, with $n$ samples and $k$ covariates in addition to genotype and intercept.
</div>