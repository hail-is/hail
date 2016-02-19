# PCA in Hail

Hail supports principal component analysis (PCA) of genotype data, a now-standard procedure ([Patterson, Price and Reich, 2006](http://journals.plos.org/plosgenetics/article?id=10.1371/journal.pgen.0020190)).

PCA expects a biallelic `.vds` file as input. The default output is a `.tsv` file recording the leading 10 principal components (PCs) of each sample. Format is one line per sample plus a header, with columns `sample PC1 PC2` etc.

Example usage:
```
$ hail read -i /path/to/input.vds pca -o /path/to/output.tsv
```

Command line options:
 - `-k <k>`, `--components <k>` -- Report the first $k$ PCs; by default $k = 10$. `<k>` can be a positive integer or `all`.
 - `-e`, `--eigenvalues` -- Write the leading $k$ sample covariance eigenvalues to `output.eigen.tsv`. Format is a single column, no header.
 - `-l`, `--loadings` -- Write the variant loadings on the leading $k$ PCs to `output.loadings.tsv`. Format is one line per variant plus a header, with columns are `chrom pos ref alt PC1 PC2` etc.

## Details

PCA is based on the singular value decomposition (SVD) of a standardized genotype matrix $M$, computed as follows. A matrix $C$ records raw biallelic genotypes, with $n$ rows indexed by samples and $m$ columns indexed by variants; $C_{ij}$ is the number of alternate alleles of variant $j$ carried by sample $i$; its values can be 0, 1, 2, or missing. For each variant $j$ the empirical minor allele frequency $p_j$ is computed as half the mean of the non-missing entries of column $j$. The corresponding column of $M$ is then mean-centered and variance-normalized as
$$
M_{ij} = \frac{C_{ij}-2p_j}{\sqrt{2p_j(1-p_j)}},
$$
with $M_{ij} = 0$ where $C_{ij}$ is missing (i.e. mean imputation). The denominator is the standard deviation under Hardy-Weinberg equilibrium (i.e. a $\mathrm{Binomial}(2,p_j)$ model) and is further motivated in the paper cited above.

PCA then computes the SVD
$$
M = USV'
$$
where columns of $U$ are the left singular vectors (orthonormal in sample space), columns of $V$ are the right singular vectors (orthonormal in variant space), and $S=\mathrm{diag}(s_1, s_2, \ldots)$ with ordered singular values $s_1 \ge s_2 \ge \cdots \ge 0$. In fact only the leading $k$ singular vectors and values are computed, yielding the best rank $k$ approximation $U_k S_k V_k'$. With option `-k all` one has $k = m$ even though there are at most $\mathrm{rank M} \le \min(m,n)$ nonzero singular values.

The default output is $MV_k = U_k S_k$, whose rows are the first $k$ principal components of each sample, i.e. the components of the projection of that sample onto the leading $k$ PCs in variant space. (N.B. This is different from PLINK, which reports $U_k$ without rescaling.)

Optionally one can also output the eigenvalues $s_1^2/m, \ldots, s_k^2/m$ of the sample covariance or genetic relatedness matrix (GRM) $MM'/m$ and the variant loadings $V_k$.

**Issue: should LD-pruning be performed first automatically or optionally once it is implemented?**
