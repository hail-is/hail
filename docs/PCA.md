# PCA in Hail

Hail supports principal component analysis (PCA) of genotype data, a now-standard procedure ([Patterson, Price and Reich, 2006](http://journals.plos.org/plosgenetics/article?id=10.1371/journal.pgen.0020190)).

As input PCA expects a `.vds` file with bialellic autosomal variants. The analysis is with respect to a standardized genotype matrix; see below for details. The default output is a `.tsv` file recording the first 10 principal component scores of each sample. Format: one line per sample plus a header, with columns `SAMPLE PC1 PC2` etc.

Example usage:
```
$ hail read -i /path/to/input.vds pca -o /path/to/output.tsv
```

Command line options:
 - `-k <k>`, `--components <k>` -- Report the first $k$ principal components; by default $k = 10$.
 - `-l`, `--loadings` -- Write the variant loadings for the first $k$ principal components to `output.loadings.tsv`. Format: one line per variant plus a header, with columns `CHROM POS REF ALT PC1 PC2` etc.
 - `-e`, `--eigenvalues` -- Write the first $k$ eigenvalues of the sample covariance or genetic relationship matrix to `output.eigen.tsv`. Format: single column, no header.


## Details

PCA is based on the singular value decomposition (SVD) of a standardized genotype matrix $M$, computed as follows. A matrix $C$ records raw genotypes, with $n$ rows indexed by samples and $m$ columns indexed by bialellic autosomal variants; $C_{ij}$ is the number of alternate alleles of variant $j$ carried by sample $i$, which can be 0, 1, 2, or missing. For each variant $j$, the sample alternate allele frequency $p_j$ is computed as half the mean of the non-missing entries of column $j$. Entries of $M$ are then mean-centered and variance-normalized as
$$
M_{ij} = \frac{C_{ij}-2p_j}{\sqrt{2p_j(1-p_j)}},
$$
with $M_{ij} = 0$ for $C_{ij}$ missing (i.e. mean genotype imputation). The normalization is under Hardy-Weinberg equilibrium (i.e. a $\mathrm{Binomial}(2, p_j)$ model) and is further motivated in the paper cited above. (The resulting amplification of signal from the low end of the allele frequency spectrum will also introduce noise for rare variants; common practice is to filter out variants with minor allele frequency below some cutoff.)

PCA then computes the SVD
$$
M = USV'
$$
where columns of $U$ are left singular vectors (orthonormal in $\mathbb{R}^n$), columns of $V$ are right singular vectors (orthonormal in $\mathbb{R}^m$), and $S=\mathrm{diag}(s_1, s_2, \ldots)$ with ordered singular values $s_1 \ge s_2 \ge \cdots \ge 0$. Typically one computes only the first $k$ singular vectors and values, yielding the best rank $k$ approximation $U_k S_k V_k'$ of $M$; the truncations $U_k$, $S_k$ and $V_k$ are $n\times k$, $k\times k$ and $m\times k$ respectively.

From the perspective of the samples or rows of $M$ as data, $V_k$ contains the variant loadings for the first $k$ principal components while $MV_k = U_k S_k$ contains the first $k$ principal component scores of the samples. The loadings represent a new basis of features while the scores represent the projected data on those features. In the output the latter are scaled by a factor of $1/\sqrt{m}$; a sample row has total variance roughly $m$ and the singular values are likewise on this order (provided $n\not\gg m$), so this scaling normalizes them to order one, independent of the number of variants.

A related object is the sample covariance or genetic relationship matrix (GRM) $MM'/m$, whose eigenvectors are the columns of $U$ and whose eigenvalues $s_1^2/m, s_2^2/m, \ldots$ are the variances carried by the respective principal components. The eigenvalues are also key statistics for the tests of population structure described in the cited paper.

**Note:** PLINK/GCTA take the GRM as starting point and compute it slightly differently with regard to missing data (modifying the denominator $m$ for each entry to the number of pairs of non-missing genotypes used). They also output its eigenvectors without rescaling (i.e. $U_k$ instead of $U_k S_k$), which has the drawback that the components no longer have the right relative variances and do not represent a projection of the original data (though the scale is immaterial for some applications such as covariates in regression).

## Issues
 - PLINK has an option to use X-chromosome variants. What is it doing exactly? There are several decisions around encoding hemizygous sites for males. More importantly, does anyone use it? Should we support it?
 - Once LD-pruning is implemented, should it be performed first automatically? My feeling is no, but the doc should mention the issue.
 - What about PCA of things other than genotypes, such as missingness? Analysts have mentioned applications to QC and flagged the latter specifically, which is implemented in GCTA.
 - What about feeding the results back into sample and variant annotations rather than writing them to files?
 - Extension to multiallelics? Few variants have more than two common alleles. Probably a one-hot encoding but variance normalization needs some care. For microsatellites/STRs a quantitative rather than categorical encoding may make sense.
