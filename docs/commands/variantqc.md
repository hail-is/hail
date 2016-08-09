<div class="cmdhead"></div>

<div class="description"></div>

<div class="synopsis"></div>

<div class="options"></div>

<div class="cmdsubsection"> 
### Notes:

 - These modules compute a variety of statistics from the genotype data, collapsing across variants.  The output from these modes can be written to TSV files, or stored within variant annotations for later use in [filtering](Filtering.md) and [export](Exporting.md). These can be accessed with `va.qc.<identifier>`.  The statistics, their types, and brief descriptions of how they are calculated are listed below.

 - Standard deviations are calculated with zero degrees of freedom.

 - Many values can be missing (`NA`).  Missingness is handled properly in filtering and is written as "NA" in export modules.

</div>

<div class="cmdsubsection">
### <a href="variantqc_annotations"> Annotations:

The below annotations can be accessed with `va.qc.<identifier>`

 - `callRate:              Double` -- Fraction of samples with called genotypes
 - `AF:                   Double` -- Calculated minor allele frequency (q)
 - `AC:                      Int` -- Count of alternate alleles
 - `rHeterozygosity:       Double` -- Proportion of heterozygotes
 - `rHetHomVar:            Double` -- Ratio of heterozygotes to homozygous alternates
 - `rExpectedHetFrequency: Double` -- Expected rHeterozygosity based on HWE
 - `pHWE:                  Double` -- p-value computed from Hardy Weinberg Equilibrium null model, [see documentation here](LeveneHaldane.tex)
 - `nHomRef:                  Int` -- Number of homozygous reference samples
 - `nHet:                     Int` -- Number of heterozygous samples
 - `nHomVar:                  Int` -- Number of homozygous alternate samples
 - `nCalled:                  Int` -- Sum of `nHomRef` + `nHet` + `nHomVar`
 - `nNotCalled:               Int` -- Number of uncalled samples
 - `nNonRef:                  Int` -- Number of het + homvar samples
 - `rHetHomVar:            Double` -- Het/HomVar ratio across all samples
 - `dpMean:                Double` -- Depth mean across all samples
 - `dpStDev:               Double` -- Depth standard deviation across all samples
 
 </div>