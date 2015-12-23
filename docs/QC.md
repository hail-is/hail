# QC in Hail

Hail includes two QC modules: `sampleqc` and `variantqc`.  These modules compute a variety of statistics from the genotype data, collapsing either variants or samples.  The output from these modes can be written to hadoop TSV files, or stored within sample and variant annotations for later use and export.  The sets of statistics, their types, and brief descriptions of how they are calculated are listed below.

**Note:** all standard deviations are calculated with zero degrees of freedom.

## Sample QC

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
 - `dpStDev;              Double` -- Depth standard deviation across all variants
 - `dpMeanHomRef:         Double` -- Depth mean across all variants called HomRef
 - `dpStDevHomRef:        Double` -- Depth standard deviation across all variants called HomRef
 - `dpMeanHet:            Double` -- Depth mean across all variants called Het
 - `dpStDevHet:           Double` -- Depth standard deviation across all variants called Het
 - `dpMeanHomVar:         Double` -- Depth mean across all variants called HomVar
 - `dpStDevHomVar:        Double` -- Depth standard deviation across all variants called HomVar
 - `gqMean:               Double` -- GQ mean across all variants
 - `gqStDev:              Double` -- GQ standard deviation across all variants
 - `gqMeanHomRef:         Double` -- GQ mean across all variants called HomRef
 - `gqStDevHomRef:        Double` -- GQ standard deviation across all variants called HomRef
 - `gqMeanHet:            Double` -- GQ mean across all variants called Het
 - `gqStDevHet:           Double` -- GQ standard deviation across all variants called Het
 - `gqMeanHomVar:         Double` -- GQ mean across all variants called HomVar
 - `gqStDevHomVar:        Double` -- GQ standard deviation across all variants called HomVar
 
## VariantQC
 - `MAF:                   Double` -- Calculated minor allele frequency (q)
 - `rHeterozygosity:       Double` -- Proportion of heterozygotes
 - `rHetHomVar:            Double` -- Ratio of heterozygotes to homozygous alternates
 - `rExpectedHetFrequency: Double` -- Expected rHeterozygosity based on HWE
 - `pHWE:                  Double` -- p-value to reject that the site is in HWE, [see documentation here](LeveneHaldane.tex)
 - `nHomRef:                  Int` -- Number of homozygous reference samples
 - `nHet:                     Int` -- Number of heterozygous samples
 - `nHomVar:                  Int` -- Number of homozygous alternate samples
 - `nCalled:                  Int` -- Sum of `nHomRef` + `nHet` + `nHomVar`
 - `nNotCalled:               Int` -- Number of uncalled samples
 - `nNonRef:                  Int` -- Number of het + homvar samples
 - `rHetHomVar:            Double` -- Het/HomVar ratio across all samples
 - `dpMean:                Double` -- Depth mean across all samples
 - `dpStDev:               Double` -- Depth standard deviation across all samples
 - `dpMeanHomRef:          Double` -- Depth mean across all samples called HomRef
 - `dpStDevHomRef:         Double` -- Depth standard deviation across all samples called HomRef
 - `dpMeanHet:             Double` -- Depth mean across all samples called Het
 - `dpStDevHet:            Double` -- Depth standard deviation across all samples called Het
 - `dpMeanHomVar:          Double` -- Depth mean across all samples called HomVar
 - `dpStDevHomVar:         Double` -- Depth standard deviation across all samples called HomVar
 - `gqMean:                Double` -- GQ mean across all samples
 - `gqStDev:               Double` -- GQ standard deviation across all samples
 - `gqMeanHomRef:          Double` -- GQ mean across all samples called HomRef
 - `gqStDevHomRef:         Double` -- GQ standard deviation across all samples called HomRef
 - `gqMeanHet:             Double` -- GQ mean across all samples called Het
 - `gqStDevHet:            Double` -- GQ standard deviation across all samples called Het
 - `gqMeanHomVar:          Double` -- GQ mean across all samples called HomVar
 - `gqStDevHomVar:         Double` -- GQ standard deviation across all samples called HomVar