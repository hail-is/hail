# PCA in Hail

Hail supports principal component analysis (PCA) of genotype data, a now-standard procedure ([Patterson, Price and Reich, 2006](http://journals.plos.org/plosgenetics/article?id=10.1371/journal.pgen.0020190)).

The procedure is based on the singular value decomposition (SVD) of a standardized genotype matrix $M$, computed as follows. A matrix $C$ records raw biallelic genotypes, with rows indexed by samples and columns indexed by variants; $C_{ij}$ is the number of alternate alleles of variant $j$ carried by sample $i$; its values can be 0, 1, 2, or missing. For each variant $j$ the empirical minor allele frequency $p_j$ is computed as half the mean of the non-missing entries of column $j$. The corresponding column of $M$ is then mean-centered and variance-normalized as
$$
M_{ij} = \frac{C_{ij}-2p_j}{\sqrt{2p_j(1-p_j)}}
$$
with $M_{ij} = 0$ where $C_{ij}$ is missing (i.e. mean imputation). The denominator is the standard deviation under Hardy-Weinberg equilibrium (i.e. a $\mathrm{Binomial}(2,p_j)$ model) and is further motivated in the paper referenced above.

PCA then computes the SVD
$$
M = USV'
$$
where columns of $U$ are the left singular vectors (orthonormal in sample space), columns of $V$ are the right singular vectors (orthonormal in variant space), and $S=\mathrm{diag}(s_1, s_2, \ldots)$ with ordered singular values $s_1 \ge s_2 \ge \cdots \ge 0$. In fact it only computes the leading $k$ singular vectors and values yielding the best rank $k$ approximation $U_k S_k V_k'$.

The default output is $MV_k = U_k S_k$, whose rows are the first $k$ principal components of each sample, i.e. the projection of that sample onto the first $k$ PCs in variant space.

One can optionally obtain also the eigenvalues $s_1^2, \ldots, s_k^2$ and the right singular vectors or variant loadings $S_k V_k$.
(Is this what we want?)

Command line arguments:
 - `-k <k>`  number of principal components to compute, default `k = 10`
 - `-var` output the SNP loadings
 - `-val` output the eigenvalues
