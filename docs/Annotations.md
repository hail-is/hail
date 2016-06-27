# Annotations in Hail

Annotations are a core aspect of data representation in Hail.  At their simplest, they are data associated with either samples or variants.  In practice, they can be thought of as SQL tables indexed by the same.  Many of Hail's modules either produce or query annotations.  

Annotations are accessed using the key prefixes for sample and variants, which are `sa` and `va` (**s**ample **a**nnotations and **v**ariant **a**nnotations).  Identifiers follow and layers of the annotatino structure are delimited by periods.  Annotations nest arbitrarily, and while most come in the form of `va.pass` or `sa.qc.callRate`, some are more complicated like `va.vep.colocated_variants[<index>].aa_maf`.

If at any time you want to know what annotations are stored within Hail, use the `showannotations` module.  This module will print the schema for both samples and variants in a readable form.

## Known annotations and their generators

Skip to:
 - [Sample QC](#sampleqc)
 - [Variant QC](#variantqc)
 - [Importing VCFs](#importvcf)
 - [VEP](#vep)
 - [Programmatic Annotation](#programmaticSamples)
 
## Sample annotations

<a name="sampleqc"></a>
#### Sample QC [(full documentation here)](QC.md#sampleqc)
##### Metrics found in `sa.qc.<identifier>`:

Identifier | Type | Description
--- | :-: | ---
`callRate` | `Double` | Fraction of variants with called genotypes
`nHomRef` | `   Int` | Number of homozygous reference variants
`nHet` | `   Int` | Number of heterozygous variants
`nHomVar` | `   Int` | Number of homozygous alternate variants
`nCalled` | `   Int` | Sum of `nHomRef` + `nHet` + `nHomVar`
`nNotCalled` | `   Int` | Number of uncalled variants
`nSNP` | `   Int` | Number of SNP variants
`nInsertion` | `   Int` | Number of insertion variants
`nDeletion` | `   Int` | Number of deletion variants
`nSingleton` | `   Int` | Number of private variants
`nTransition` | `   Int` | Number of transition (A-G, C-T) variants
`nTransversion` | `   Int` | Number of transversion variants
`nNonRef` | `   Int` | Number of Het + HomVar variants
`rTiTv` | `Double` | Transition/transversion ratio
`rHetHomVar` | `Double` | Het/HomVar ratio across all variants
`rDeletionInsertion` | `Double` | Deletion/Insertion ratio across all variants
`dpMean` | `Double` | Depth mean across all variants
`dpStDev` | `Double` | Depth standard deviation across all variants
`gqMean` | `Double` | GQ mean across all variants
`gqStDev` | `Double` | GQ standard deviation across all variants
 
____
 
Additionally, users can import annotations from a variety of files, or define new annotations as functions of the exposed data structures.  [See the full documentation here.](ImportAnnotations.md) 
  
## Variant annotations

<a name="variantqc"></a>
#### Variant QC [(full documentation here)](QC.md#variantqc)
##### Metrics found in `va.qc.<identifier>`:

Identifier | Type | Description
--- | :-: | ---
`callRate` | `Double` | Fraction of samples with called genotypes
`AF` | `Double` | Calculated alternate allele frequency (q)
`AC` | `   Int` | Count of alternate alleles
`rHeterozygosity` | `Double` | Proportion of heterozygotes
`rHetHomVar` | `Double` | Ratio of heterozygotes to homozygous alternates
`rExpectedHetFrequency` | `Double` | Expected rHeterozygosity based on HWE
`pHWE` | `Double` | p-value computed from Hardy Weinberg Equilibrium null model, [see documentation here](LeveneHaldane.tex)
`nHomRef` | `   Int` | Number of homozygous reference samples
`nHet` | `   Int` | Number of heterozygous samples
`nHomVar` | `   Int` | Number of homozygous alternate samples
`nCalled` | `   Int` | Sum of `nHomRef` + `nHet` + `nHomVar`
`nNotCalled` | `   Int` | Number of uncalled samples
`nNonRef` | `   Int` | Number of het + homvar samples
`rHetHomVar` | `Double` | Het/HomVar ratio across all samples
`dpMean` | `Double` | Depth mean across all samples
`dpStDev` | `Double` | Depth standard deviation across all samples
`gqMean` | `Double` | GQ mean across all samples
`gqStDev` | `Double` | GQ standard deviation across all samples
 
____
 
<a name="importvcf"></a>
#### Imported VCFs [(full documentation here)](Importing.md#annotations)
##### Metrics found in `va.<identifier>`:

Identifier | Type | Description
--- | :-: | ---
`pass` | `Boolean` | true if the variant contains `PASS` in the filter field (false if `.` or other)
`filters`  | `Set[String]` | set containing the list of filters applied to a variant.  Accessible using `va.filters.contains("VQSRTranche99.5...")`, for example
`rsid`          | `String` | rsid of the variant, if it has one ("." otherwise)
`qual`          | `Double` | the number in the qual field
`multiallelic` | `Boolean` | true if the variant is multiallelic or was split
`info.<field>`       | `T` | matches (with proper capitalization) any defined info field.  Data types match the type specified in the vcf header, and if the `Number` is "A", "R", or "G", the result will be stored in an array (accessed with array\[index\]).
 
____
 
<a name="splitmulti"></a>
#### Splitting multiallelics [(full documentation here)](Splitmulti.md#annotations)
##### Metrics found in `va.<identifier>`:

Identifier | Type | Description
--- | :-: | ---
`wasSplit`       | `Boolean` | this variant was originally multiallelic 
`aIndex`         | `    Int` | The original index of this variant in imported line, 0 for imported biallelic sites

____

<a name="vep"></a>
#### Variant effect predictor (VEP)  
##### Metrics found in `va.vep.<identifier>`.  Schema is too long to reproduce here, see [VEP documentation](VEP.md#annotations) for full schema

____

<a name="programmaticVariants"></a>
#### Programmatic annotation [(full documentation here)](ProgrammaticAnnotation.md)
##### Found in any path, follow the previous link for details.