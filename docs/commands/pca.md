<div class="cmdhead"></div>

<div class="description"></div>

<div class="synopsis"></div>

<div class="options"></div>

<div class="cmdsubsection">
### Implementation Details:

Hail supports principal component analysis (PCA) of genotype data, a now-standard procedure ([Patterson, Price and Reich, 2006](http://journals.plos.org/plosgenetics/article?id=10.1371/journal.pgen.0020190)).

PCA is based on the singular value decomposition (SVD) of a standardized genotype matrix $M$, computed as follows.

An $n\times m$ matrix $C$ records raw genotypes, with rows indexed by $n$ samples and columns indexed by $m$ bialellic autosomal variants; $C_{ij}$ is the number of alternate alleles of variant $j$ carried by sample $i$, which can be 0, 1, 2, or missing. For each variant $j$, the sample alternate allele frequency $p_j$ is computed as half the mean of the non-missing entries of column $j$. Entries of $M$ are then mean-centered and variance-normalized as
$$
M_{ij} = \frac{C_{ij}-2p_j}{\sqrt{2p_j(1-p_j)m}},
$$
with $M_{ij} = 0$ for $C_{ij}$ missing (i.e. mean genotype imputation). This scaling normalizes genotype variances to a common value $1/m$ for variants in Hardy-Weinberg equilibrium and is further motivated in the paper cited above. (The resulting amplification of signal from the low end of the allele frequency spectrum will also introduce noise for rare variants; common practice is to filter out variants with minor allele frequency below some cutoff.)  The factor $1/m$ gives each sample row approximately unit total variance (assuming linkage equilibrium) and yields the sample correlation or genetic relationship matrix (GRM) as simply $MM^T$.

PCA then computes the SVD
$$
M = USV^T
$$
where columns of $U$ are left singular vectors (orthonormal in $\mathbb{R}^n$), columns of $V$ are right singular vectors (orthonormal in $\mathbb{R}^m$), and $S=\mathrm{diag}(s_1, s_2, \ldots)$ with ordered singular values $s_1 \ge s_2 \ge \cdots \ge 0$. Typically one computes only the first $k$ singular vectors and values, yielding the best rank $k$ approximation $U_k S_k V_k^T$ of $M$; the truncations $U_k$, $S_k$ and $V_k$ are $n\times k$, $k\times k$ and $m\times k$ respectively.

From the perspective of the samples or rows of $M$ as data, $V_k$ contains the variant loadings for the first $k$ PCs while $MV_k = U_k S_k$ contains the first $k$ PC scores of each sample. The loadings represent a new basis of features while the scores represent the projected data on those features. The eigenvalues of the GRM $MM^T$ are the squares of the singular values $s_1^2, s_2^2, \ldots$, which represent the variances carried by the respective PCs.

**Note:** In PLINK/GCTA the GRM is taken as the starting point and it is computed slightly differently with regard to missing data. Here the $ij$ entry of $MM^T$ is simply the dot product of rows $i$ and $j$ of $M$; in terms of $C$ it is
$$
\frac{1}{m}\sum_{l\in\mathcal{C}_i\cap\mathcal{C}_j}\frac{(C_{il}-2p_l)(C_{jl} - 2p_l)}{2p_l(1-p_l)}
$$
where $\mathcal{C}_i = \{l\mid C_{il}\text{ is non-missing}\}$. In PLINK/GCTA the denominator $m$ is replaced with the number of terms in the sum $\lvert\mathcal{C}_i\cap\mathcal{C}_j\rvert$, i.e. the number of variants where both samples have non-missing genotypes. While this is arguably a better estimator of the true GRM (trading shrinkage for noise), it has the drawback that one loses the clean interpretation of the loadings and scores as features and projections.

Separately, for the PCs PLINK/GCTA output the eigenvectors of the GRM; even ignoring the above discrepancy that means the left singular vectors $U_k$ instead of the component scores $U_k S_k$. While this is just a matter of the scale on each PC, the scores have the advantage of representing true projections of the data onto features with the variance of a score reflecting the variance explained by the corresponding feature. (In PC bi-plots this amounts to a change in aspect ratio; for use of PCs as covariates in regression it is immaterial.)
</div>

<div class="cmdsubsection">
### Examples:

As input, PCA expects a `.vds` file with biallelic autosomal variants. The analysis is with respect to a standardized genotype matrix; see below for details. The default output is a `.tsv` file recording the first 10 principal component (PC) scores of each sample. Format: one line per sample plus a header, with columns `SAMPLE`, `PC1`, `PC2`, etc.

Example usage:
```
$ hail read -i /path/to/file.vds pca -o /path/to/file.tsv
```
</div>
