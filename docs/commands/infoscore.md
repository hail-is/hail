<div class="cmdhead"></div>

<div class="description"></div>

<div class="synopsis"></div>

<div class="options"></div>

<div class="cmdsubsection">
### Implementation Details

We implemented the IMPUTE info measure as described in the [supplementary information from Marchini & Howie. Genotype imputation for genome-wide association studies. Nature Reviews Genetics (2010)](http://www.nature.com/nrg/journal/v11/n7/extref/nrg2796-s3.pdf).

#### Algorithm

To calculate the info score $I_{A}$ for one SNP:

$$
I_{A} = 
\begin{cases}
1 - \frac{\sum_{i=1}^{N}(f_{i} - e_{i}^2)}{2N\hat{\theta}(1 - \hat{\theta})} & \text{when } \hat{\theta} \in (0, 1) \\
1 & \text{when } \hat{\theta} = 0, \hat{\theta} = 1\\
\end{cases}
$$

 - $N$ is the number of samples with imputed genotype probabilities [$p_{ik} = P(G_{i} = k)$ where $k \in \{0, 1, 2\}$]
 - $e_{i} = p_{i1} + 2p_{i2}$ is the expected genotype per sample
 - $f_{i} = p_{i1} + 4p_{i2}$
 - $\hat{\theta} = \frac{\sum_{i=1}^{N}e_{i}}{2N}$ is the MLE for the population minor allele frequency

#### Consistency of results with qctool
Hail will not generate identical results as [qctool](http://www.well.ox.ac.uk/~gav/qctool/#overview) for the following reasons:
 
 - The floating point number Hail stores for each dosage is slightly different than the original data due to rounding and normalization of probabilities.
 - Hail automatically removes dosages that [do not meet certain requirements](#dosagefilters) on data import with [`importgen`](#importgen) and [`importbgen`](#importbgen).
 - Hail does not use the population frequency to impute dosages when a dosage has been set to missing.
 - Hail calculates the same statistic for sex chromosomes as autosomes while qctool incorporates sex information

**The info score Hail reports will be extremely different than qctool when a SNP has a high missing rate.**

</div>

<div class="cmdsubsection">
### Annotations

The below annotations can be accessed with `va.infoscore.<identifier>`

Identifier | Type | Description
--- | :-: | ---
`impute` | `Double` | IMPUTE info score
`nIncluded` | `Int` | Number of samples with non-missing dosages

</div>

<div class="cmdsubsection">
### Examples

<h4 class="example">Calculate info scores from dosage data and export annotations to file</h4>
```
hail importgen -s /my/path/example.sample /my/path/example.gen 
    infoscore 
    exportvariants -c 'v, va.infoscore.impute, va.infoscore.nIncluded' 
                   -o infoScores.tsv
```

</div>