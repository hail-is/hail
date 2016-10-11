<div class="cmdhead"></div>

<div class="description"></div>

<div class="synopsis"></div>

<div class="options"></div>

<div class="cmdsubsection">
### Implementation Details

The `logreg` command performs, for each variant, a significance test of the genotype in predicting a binary (case-control) phenotype based on the logistic regression model. Hail supports the Wald test, likelihood ratio test (LRT), and Rao score test. Hail only includes samples for which phenotype and all covariates are defined. For each variant, Hail imputes missing genotypes as the mean of called genotypes.

Assuming there are sample annotations `sa.pheno.isCase`, `sa.cov.age`, `sa.cov.isFemale`, and `sa.cov.PC1`, the command
```
logreg -y sa.pheno.isCase -c sa.cov.age,sa.cov.isFemale,sa.cov.PC1
```
considers a model of the form
$$
\mathrm{Prob}(\mathrm{isCase}) = \mathrm{sigmoid}(\beta_0 + \beta_1 \, \mathrm{gt} + \beta_2 \, \mathrm{age} + \beta_3 \, \mathrm{isMale} + \beta_4 \, \mathrm{PC1} + \varepsilon), \quad \varepsilon \sim \mathrm{N}(0, \sigma^2)
$$
where $\mathrm{sigmoid}$ is the [sigmoid function](https://en.wikipedia.org/wiki/Sigmoid_function), the genotype $\mathrm{gt}$ is coded as $0$ for HomRef, $1$ for Het, and $2$ for HomVar, and the Boolean covariate $\mathrm{isFemale}$ is coded as $1$ for true (female) and $0$ for false (male). The null model sets $\beta_1 = 0$.

The resulting variant annotations depend on the test statistic as shown in the tables below. These annotations can then be accessed by other methods, including exporting to TSV with other variant annotations.

Test | Annotation | Type | Value
---|---|---|---
Wald | `va.logreg.wald.beta` | Double | fit genotype coefficient, $\hat\beta_1$
Wald | `va.logreg.wald.se` | Double | estimated standard error, $\widehat{\mathrm{se}}$ 
Wald | `va.logreg.wald.zstat` | Double | Wald $z$-statistic, equal to $\hat\beta_1 / \widehat{\mathrm{se}}$
Wald | `va.logreg.wald.pval` | Double | Wald test $p$-value testing $\beta_1 = 0$
LRT | `va.logreg.lrt.beta` | Double | fit genotype coefficient, $\hat\beta_1$
LRT | `va.logreg.lrt.chi2` | Double | likelihood ratio test statistic (deviance) testing $\beta_1 = 0$
LRT | `va.logreg.lrt.pval` | Double | likelihood ratio test $p$-value
Score | `va.logreg.score.chi2` | Double | score statistic testing $\beta_1 = 0$
Score | `va.logreg.score.pval` | Double | score test $p$-value

For the Wald and likelihood ratio tests, Hail fits the logistic model for each variant using Newton iteration and only emits the above annotations when the maximum likelihood estimate of the coefficients converges. To help diagnose convergence issues, Hail also emits three variant annotations which summarize the iterative fitting process:

Test | Annotation | Type | Value
---|---|---|---
Wald, LRT | `va.logreg.fit.nIter` | Int | number of iterations until convergence, explosion, or reaching the max (25)
Wald, LRT | `va.logreg.fit.converged` | Boolean | true if iteration converged
Wald, LRT | `va.logreg.fit.exploded` | Boolean | true if iteration exploded

We consider iteration to have converged when every coordinate of $\beta$ changes by less than $10^{-6}$. Up to 25 iterations are attempted; in testing we find 4 or 5 iterations nearly always suffice. Convergence may also fail due to explosion, which refers to low-level numerical linear algebra exceptions caused by manipulating ill-conditioned matrices. Explosion may result from (nearly) linearly dependent covariates or complete [separation](https://en.wikipedia.org/wiki/Separation_(statistics)).

A more common situation in genetics is quasi-complete seperation, e.g. variants that are observed only in cases (or controls). Such variants inevitably arise when testing millions of variants with very low minor allele count. The maximum likelihood estimate of $\beta$ under logistic regression is then undefined but convergence may still occur after a large number of iterations due to a very flat likelihood surface. In testing, we find that such variants produce a secondary bump from 10 to 15 iterations in the histogram of number of iterations per variant. We also find that this faux convergence produces large standard errors and large (insignificant) $p$-values. To not miss such variants, consider using Firth logistic regression, linear regression, or group-based tests. 

Here's a concrete illustration of quasi-complete seperation in R. Suppose we have 2010 samples distributed as follows for a particular variant:

\- | HomRef | Het | HomVar
---|---|---|---
Case | 1000 | 10 | 0
Control | 1000 | 0 | 0

The following R code fits the (standard) logistic, Firth logistic, and linear regression models to this data, where $x$ is genotype, $y$ is phenotype, and `logistf` is from the `logistf` package.
```
x <- c(rep(0,1000), rep(1,1000), rep(1,10)
y <- c(rep(0,1000), rep(0,1000), rep(1,10))
logfit <- glm(y ~ x, family=binomial())
firthfit <- logistf(y ~ x)
linfit <- lm(y ~ x)
```
The resulting $p$-values for the genotype coefficient are $0.991$, $0.00085$, and $0.0016$, respectively. The erroneous value $0.991$ is due to quasi-complete separation. Moving one of the 10 hets from case to control eliminates this quasi-complete separation; the p-values from R are then $0.0373$, $0.0111$, and $0.0116$, respectively, as expected for a less significant association.

Phenotype and covariate sample annotations may also be specified using [programmatic expressions](reference.html#HailExpressionLanguage) without identifiers, such as
```
if (sa.isFemale) sa.cov.age else (2 * sa.cov.age + 10)
```
For Boolean covariate types, true is coded as 1 and false as 0. In particular, for the sample annotation `sa.fam.isCase` added by importing a `.fam` file with case-control phenotype, case is $1$ and control is $0$.

Hail's logistic regression tests correspond to the `b.wald`, `b.lrt`, and `b.score` tests in [EPACTS](http://genome.sph.umich.edu/wiki/EPACTS#Single_Variant_Tests). For each variant, Hail imputes missing genotypes as the mean of called genotypes, whereas EPACTS subsets to those samples with called genotypes. Hence, Hail and EPACTS results will currently only agree for variants with no missing genotypes.

See [Recommended joint and meta-analysis strategies for case-control association testing of single low-count variants](http://www.ncbi.nlm.nih.gov/pmc/articles/PMC4049324/) for an empirical comparison of the logistic Wald, LRT, score, and Firth tests. The theoretical foundations of the Wald, likelihood ratio, and score tests may be found in Chapter 3 of Gesine Reinert's notes [Statistical Theory](http://www.stats.ox.ac.uk/~reinert/stattheory/theoryshort09.pdf).
</div>
