<div class="cmdhead"></div>

<div class="description"></div>

<div class="synopsis"></div>

<div class="options"></div>

<div class="cmdsubsection">
### Implementation Details

The `lmmreg` command estimates the genetic proportion of residual phenotypic variance (narrow-sense heritability) under a kinship-based linear mixed model, and then (optionally) tests each variant for association using the likelihood ratio test. Inference is exact.

#### Using the command

Assuming there are numeric or Boolean sample annotations `sa.pheno`, `sa.cov1`, `sa.cov2` and Boolean variant annotations `va.inKinship` and `va.inAssoc`, the command
```
lmmreg \
  -y sa.pheno \
  -c sa.cov1,sa.cov2 \
  -k va.inKinship \
  -a va.inAssoc
```
will execute the following five steps in order:

1) filter to samples for which `sa.pheno`, `sa.cov`, and `sa.cov2` are all defined
2) compute the kinship matrix $K$ (the [RRM](#kinship_RRM) defined below) using those variants for which `va.inKinship` is true
3) compute the eigendecomposition $K = USU^T$ of the kinship matrix
4) fit covariate coefficients and variance parameters in the sample-covariates-only (global) model using restricted maximum likelihood ([REML](https://en.wikipedia.org/wiki/Restricted_maximum_likelihood)), storing results in global annotations under `global.lmmreg`
5) test each variant with `va.inAssoc` true for association, storing results under `va.lmmreg` in variant annotations

More generally, both `-k` (`--kinshipfilt`) and `-a` (`--assocfilt`) take arbitrary Boolean-valued variant annotation expressions, so for example one could write

```
-a 'v.isAutosomal && va.qc.AF > .01 && va.qc.AF < .99'
```

This plan can be modified as follows:

- Remove the `-a` (`--assocfilter`) option to test *all* variants for association in Step 5.
- Add the `-n` (`--noassoc`) flag to test *no* variants for association, i.e. skip Step 5.
- Add the `--useml` flag to use maximum likelihood instead of REML in Steps 4 and 5.
- Use the `--delta` option to manually set the value of $\delta$ rather that fitting $\delta$ in Step 4.
- Use the `--globalroot` option to change the global annotation root in Step 4.
- Use the `--varoot` option to change the variant annotation root in Step 5.

`lmmreg` adds eight global annotations in Step 4; the last three are omitted if the `--delta` option is used.

Annotation | Type | Value
---|---|---
`global.lmmreg.useML` | Boolean | true if fit by ML, false if fit by REML
`global.lmmreg.beta` | Dict[String, Double] | Map from "intercept" and the given `--cov` expressions to the corresponding fit $\beta$ coefficients
`global.lmmreg.sigmaG2` | Double | fit coefficient of genetic variance, $\hat{\sigma}_g^2$
`global.lmmreg.sigmaE2` | Double | fit coefficient of environmental variance $\hat{\sigma}_e^2$
`global.lmmreg.delta` | Double | fit ratio of variance component coefficients, $\hat{\delta}$
`global.lmmreg.h2` | Double | fit narrow-sense heritability, $\hat{h}^2
`global.lmmreg.evals` | Array[Double] | eigenvalues of the kinship matrix in descending order
`global.lmmreg.logDeltaGrid` | Array[Double] | values of $\mathit{ln}(\delta)$ used in the grid search
`global.lmmreg.logLkhdVals` | Array[Double] | (restricted) log likelihood of $y$ given $X$ and $\mathit{ln}(\delta)$ at the (RE)ML fit of $\beta$ and $\sigma_g^2$
`global.lmmreg.maxLogLkhd` | Double | (restricted) maximum log likelihood corresponding to the fit delta

These global annotations are also added to `hail.log`, with the ranked evals and $\delta$ grid with values in `tsv` tabular form.  Use `grep 'lmmreg:' hail.log` to find the lines just above each table.

If Step 5 is performed, `lmmreg` also adds nine variant annotations.

Annotation | Type | Value
---|---|---
`va.lmmreg.beta` | Double | fit genotype coefficient, $\hat\beta_0$
`va.lmmreg.sigmaG2` | Double | fit coefficient of genetic variance component, $\hat{\sigma}_g^2$
`va.lmmreg.chi2` | Double | $\chi^2$ statistic of the likelihood ratio test
`va.lmmreg.pval` | Double | $p$-value
`va.lmmreg.maf` | Double | minor allele frequency for included samples
`va.lmmreg.nHomRef` | Int | count of HomRef genotypes for included samples
`va.lmmreg.nHet` | Int | count of Het genotypes among for samples
`va.lmmreg.nHomVar` | Int | count of HomVar genotypes for included samples
`va.lmmreg.nMissing` | Int | count of missing genotypes for included samples

Unlike the `filtervariants` command, the `-k` (`--kinshipfilter`) and `-a` (`--assocfilter`) filters do *not* remove variants from the underlying variant dataset.  Rather, variants for which `va.inAssoc` is false will have missing values for these annotations, as will those variants that don't vary across the included samples (e.g., all genotypes are HomRef). If both the kinship variants and association variants reside is a subset of the genome, consider running `filtervariants` before `lmmreg`.  You can also `filtervariants` after running `lmmreg` but before running `exportvariants` to avoid outputting variants with missing annotations.  The simplest way to export all resulting annotations is:

```
  showglobals \
    -o lmmreg.json \
  exportvariants \
    -c 'variant = v, va.lmmreg.*' \
    -o lmmreg.tsv.bgz
```

Similarly `lmmreg` does *not* remove samples from the underlying variant dataset.

#### Performance

This initial version of `lmmreg` scales to tens of thousands samples and an unbounded number of variants, making it particularly well-suited to modern sequencing studies. For example, starting from a VDS of the 1000 Genomes Project (consisting of 2535 whole genomes), `lmmreg` computes a kinship matrix based on 100k common variants, fits coefficients and variance components in the sample-covariates-only model, runs a linear-mixed-model likelihood ratio test for all 15 million high-quality non-rare variants, and exports the results in 3m42s minutes. Here we used 42 preemptible workers (~680 cores) on 2k partitions at a compute cost of about 50 cents on Google cloud (see [Using Hail on the Google Cloud Platform](http://discuss.hail.is/t/using-hail-on-the-google-cloud-platform/80)). The first analysts to apply `lmmreg` in research computed kinship from 262k common variants and tested 25 million non-rare variants on 8185 whole genomes in 32 minutes.

While `lmmreg` computes the kinship matrix $K$ using distributed matrix multiplication (Step 2), the full eigendecomposition (Step 3) is currently run on a single core of master using the [LAPACK routine DSYEVD](http://www.netlib.org/lapack/explore-html/d2/d8a/group__double_s_yeigen_ga694ddc6e5527b6223748e3462013d867.html), which we empirically find to be the most performant of the four available routines; laptop performance plots showing cubic complexity in $n$ are available [here](https://github.com/hail-is/hail/pull/906). On Google cloud, eigendecomposition takes about 2 seconds for 2535 sampes and 1 minute for 8185 samples. If you see worse performance, check that LAPACK natives are being properly loaded (see "BLAS and LAPACK" in [Getting Started](https://hail.is/getting_started.html)). We plan to add functionality to save intermediate results so they need not be recalculated when following up on genomic regions of interest.

Given the eigendecomposition, fitting the global model (Step 4) takes on the order of a few seconds on master. Association testing (Step 5) is fully distributed by variant with per-variant time complexity that is completely independent of the number of sample covariates and dominated by multiplication of the genotype vector $v$ by the matrix of eigenvectors $U^T$ as described below, which we accelerate with a sparse representation of $v$.  The matrix $U^T$ has size about $8n^2$ bytes and is currently broadcast to each Spark executor. For example, with 15k samples, storing $U^T$ consumes about 3.6GB of memory on a 16-core worker node with two 8-core executors. So for large $n$, we recommend using a high-memory configuration such as `highmem` workers.

#### Linear mixed model

We first describe the sample-covariates-only model used to estimate heritability, which we simply refer to as the **global model**. With $n$ samples and $c$ sample covariates, we define:

- $y = n \times 1$ vector of phenotypes
- $X = n \times c$ matrix of sample covariates and intercept column of ones
- $K = n \times n$ kinship matrix
- $I = n \times n$ identity matrix
- $\beta = c \times 1$ vector of covariate coefficients
- $\sigma_g^2 =$ coefficient of genetic variance component $K$
- $\sigma_e^2 =$ coefficient of environmental variance component $I$
- $\delta = \frac{\sigma_e^2}{\sigma_g^2} =$ ratio of environmental and genetic variance component coefficients
- $h^2 = \frac{\sigma_g^2}{\sigma_g^2 + \sigma_e^2} = \frac{1}{1 + \delta} =$ genetic proportion of residual phenotypic variance

Under a linear mixed model, $y$ is sampled from the $n$-dimensional [multivariate normal distribution](https://en.wikipedia.org/wiki/Multivariate_normal_distribution) with mean $X \beta$ and variance components that are scalar multiples of $K$ and $I$:

$$y \sim \mathrm{N}\left(X\beta, \sigma_g^2 K + \sigma_e^2 I\right)$$

Thus the model posits that the residuals $y_i - X_{i,:}\beta$ and $y_j - X_{j,:}\beta$ have covariance $\sigma_g^2 K_{ij}$ and approximate correlation $h^2 K_{ij}$. Informally: phenotype residuals are correlated as the product of overall heritability and pairwise kinship. By contrast, standard (unmixed) linear regression is equivalent to fixing $\sigma_2$ (equivalently, $h^2$) at $0$ above, so that all phenotype residuals are independent.

A word of caution: while it is tempting to interpret $h^2$ as the [narrow-sense heritability](https://en.wikipedia.org/wiki/Heritability#Definition) of the phenotype alone, note that its value depends not only the phenotype and genetic data, but also on the choice of sample covariates.

#### Fitting the global model

The core of `lmmreg` is a distributed implementation of the spectral approach taken in [FastLMM](https://www.microsoft.com/en-us/research/project/fastlmm/). Let $K = USU^T$ be the [eigendecomposition](https://en.wikipedia.org/wiki/Eigendecomposition_of_a_matrix#Real_symmetric_matrices) of the real symmetric matrix $K$. That is:

- $U = n \times n$ orthonormal matrix whose columns are the eigenvectors of $K$
- $S = n \times n$ diagonal matrix of eigenvalues of $K$ in descending order. $S_{ii}$ is the eigenvalue of eigenvector $U_{:,i}$
- $U^T = n \times n$ orthonormal matrix, the transpose (and inverse) of $U$

A bit of matrix algebra on the multivariate normal density shows that the linear mixed model above is mathematically equivalent to the model

$$U^Ty \sim \mathrm{N}\left(U^TX\beta, \sigma_g^2 (S + \delta I)\right)$$

for which the covariance is diagonal (e.g., unmixed). That is, rotating the phenotype vector ($y$) and covariate vectors (columns of $X$) in $\mathbb{R}^n$ by $U^T$ transforms the model to one with independent residuals. For any particular value of $\delta$, the restricted maximum likelihood (REML) solution for the latter model can be solved exactly in time complexity that is linear rather than cubic in $n$.  In particular, having rotated, we can run a very efficient 1-dimensional optimization procedure over $\delta$ to find the REML estimate $(\hat{\delta}, \hat{\beta}, \hat{\sigma}_g^2)$ of the triple $(\delta, \beta, \sigma_g^2)$, which in turn determines $\hat{\sigma}_e^2$ and $\hat{h}^2$.

We first compute the maximum log likelihood on a $\delta$-grid that is uniform on the log scale, with $\mathit{ln}(\delta)$ running from $-10$ to $10$ by $0.01$, corresponding to $h^2$ decreasing from $0.999999998$ to $0.000000002$. If $h^2$ is maximized at the lower boundary then standard linear regression would be more appropriate and Hail will exit; more generally, consider using standard linear regression when $\hat{h}^2$ is very small. A maximum at the upper boundary is highly suspicious and will also cause Hail to exit, with the `hail.log` recording all values over the grid for further inspection.

If the optimal grid point falls in the interior of the grid as expected, we then use [Brent's method](https://en.wikipedia.org/wiki/Brent%27s_method) to find the precise location of the maximum over the same range, with initial guess given by the optimal grid point and a tolerance on $\mathit{ln}(\delta)$ of $1e-6$. If this location differs from the optimal grid point by more than $.01$, a warning will be displayed and logged, and one would be wise to investigate by plotting the values over the grid. Note that $h^2$ is related to $\mathit{ln}(\delta)$ through the [sigmoid function](https://en.wikipedia.org/wiki/Sigmoid_function). Hence one can change variables to extract a high-resolution discretization of the likelihood function of $h^2$ over $[0,1]$ at the corresponding REML estimators for $\beta$ and $\sigma_g^2$.

#### Testing each variant for association

Fixing a single variant, we define:

- $v = n \times 1$ vector of genotypes, with missing genotypes imputed as the mean of called genotypes
- $X_v = \left[v | X \right] = n \times (1 + c)$ matrix concatenating $v$ and $X$
- $\beta_v = (\beta^0_v, \beta^1_v, \ldots, \beta^c_v) = (1 + c) \times 1$ vector of covariate coefficients

Fixing $\delta$ at the global REML estimate $\hat{\delta}$, we find the REML estimate $(\hat{\beta}_v, \hat{\sigma}_{g,v}^2)$ via rotation of the model

$$y \sim \mathrm{N}\left(X_v\beta_v, \sigma_{g,v}^2 (K + \hat{\delta} I)\right)$$

Note that the only new rotation to compute here is $U^T v$.

To test the null hypothesis that the genotype coefficient $\beta^0_v$ is zero, we consider the restricted model with parameters $((0, \beta^1_v, \ldots, \beta^c_v), \sigma_{g,v}^2)$ within the full model with parameters $(\beta^0_v, \beta^1_v, \ldots, \beta^c_v), \sigma_{g_v}^2)$, with $\delta$ fixed at $\hat\delta$ in both. The latter fit is simply that of the global model, $((0, \hat{\beta}^1, \ldots, \hat{\beta}^c), \hat{\sigma}_g^2)$. The likelihood ratio test statistic is given by

$$\chi^2 = n \, \mathit{ln}\left(\frac{\hat{\sigma}^2_g}{\hat{\sigma}_{g,v}^2}\right)$$

and follows a chi-squared distribution with one degree of freedom. Here the ratio $\hat{\sigma}^2_g / \hat{\sigma}_{g,v}^2$ captures the degree to which adding the variant $v$ to the global model reduces the residual phenotypic variance.

#### <a class="jumptarget" name="kinship_RRM"></a> Kinship: Realized Relationship Matrix (RRM)

As in FastLMM, `lmmreg` uses the Realized Relationship Matrix (RRM) for kinship, defined as follows. Consider the $n\times m$ matrix $C$ of raw genotypes, with rows indexed by $n$ samples and columns indexed by the $m$ bialellic autosomal variants for which `va.inKinship` is true; $C_{ij}$ is the number of alternate alleles of variant $j$ carried by sample $i$, which can be 0, 1, 2, or missing. For each variant $j$, the sample alternate allele frequency $p_j$ is computed as half the mean of the non-missing entries of column $j$. Entries of $M$ are then mean-centered and variance-normalized as

$$ M_{ij} = \frac{C_{ij}-2p_j}{\sqrt{\frac{m}{n} \sum_{k=1}^n (C_{ij}-2p_j)^2}}, $$

with $M_{ij} = 0$ for $C_{ij}$ missing (i.e. mean genotype imputation). This scaling normalizes each variant column to have empirical variance $1/m$, which gives each sample row approximately unit total variance (assuming linkage equilibrium) and yields the $n \times n$ sample correlation or realized relationship matrix (RRM) $K$ as simply $$K = MM^T$$ Note that the only difference between the Realized Relationship Matrix and the Genetic Relationship Matrix (GRM) used in the [PCA command](https://hail.is/commands.html#pca) is the variant (column) normalization: where RRM uses empirical variance, GRM uses expected variance under Hardy-Weinberg Equilibrium.

#### Further background
For the history and mathematics of linear mixed models in genetics, including [FastLMM](https://www.microsoft.com/en-us/research/project/fastlmm/), see [Christoph Lippert's PhD thesis](https://publikationen.uni-tuebingen.de/xmlui/bitstream/handle/10900/50003/pdf/thesis_komplett.pdf). For an investigation of various approaches to defining kinship, see [Comparison of Methods to Account for Relatedness in Genome-Wide Association Studies with Family-Based Data](http://journals.plos.org/plosgenetics/article?id=10.1371/journal.pgen.1004445).

</div>