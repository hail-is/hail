<div class="cmdhead"></div>

<div class="description"></div>

<div class="synopsis"></div>

<div class="options"></div>

<div class="cmdsubsection">
### Notes:

 - These modules compute a variety of statistics from the genotype data, collapsing across samples.  The output from these modes can be written to TSV files, or stored within sample annotations for later use in [filtering](Filtering.md) and [export](Exporting.md). These can be accessed with `sa.qc.<identifier>`.  The statistics, their types, and brief descriptions of how they are calculated are listed below.

 - All standard deviations are calculated with zero degrees of freedom.

 - Many values can be missing (`NA`).  Missingness is handled properly in filtering and is written as "NA" in export modules.
</div>


<div class="cmdsubsection">
### <a href="sampleqc_annotations"></a> Annotations:

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
 
 </div>