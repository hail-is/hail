# Quality Control

Hail includes two QC modules: 
 - `sampleqc`
 - `variantqc`
 
These modules compute a variety of statistics from the genotype data, collapsing either variants or samples.  The output from these modes can be written to TSV files, or stored within sample and variant annotations for later use in [filtering](Filtering.md) and [export](Exporting.md).  
  
Command line option:
 - `-o | --output <filename>` -- **(Optional)** an output file path
 
All computed statistics will be stored within Hail in sample or variant annotations, to be used with downstream [filter](Filtering.md) and [export](Exporting.md) modules.  These can be accessed with `va.qc.<identifier>` and `sa.qc.<identifier>` for variant and sample qc, respectively.  The statistics, their types, and brief descriptions of how they are calculated are listed below.

**Note:** All standard deviations are calculated with zero degrees of freedom.

**Note:** Many values can be missing (`NA`).  Missingness is handled properly in filtering and is written as "NA" in export modules.

<a name="sampleqc"></a>
## Sample QC

The below annotations can be accessed with `sa.qc.<identifier>`

 - `callRate:             Double` -- Fraction of variants with called genotypes
 - `nHomRef:                 Int` -- Number of homozygous reference variants
 - `nHet:                    Int` -- Number of heterozygous variants
 - `nHomVar:                 Int` -- Number of homozygous alternate variants
 - `nCalled:                 Int` -- Sum of `nHomRef` + `nHet` + `nHomVar`
 - `nNotCalled:              Int` -- Number of uncalled variants
 - `nSNP:                    Int` -- Number of SNP variants
 - `nInsertion:              Int` -- Number of insertion variants
 - `nDeletion:               Int` -- Number of deletion variants
 - `nSingleton:              Int` -- Number of private variants
 - `nTransition:             Int` -- Number of transition (A-G, C-T) variants
 - `nTransversion:           Int` -- Number of transversion variants
 - `nNonRef:                 Int` -- Number of Het + HomVar variants
 - `rTiTv:                Double` -- Transition/transversion ratio
 - `rHetHomVar:           Double` -- Het/HomVar ratio across all variants
 - `rDeletionInsertion:   Double` -- Deletion/Insertion ratio across all variants    
 - `dpMean:               Double` -- Depth mean across all variants
 - `dpStDev:              Double` -- Depth standard deviation across all variants
 
<a name="variantqc"></a>
## VariantQC

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