# `annotatevariants`

This module is a supercommand that contains four submodules:

Name | Docs | Description
:-:  | :-: | ---
`annotatevariants table` | [**\[link\]**](AnnotateVariantsTable.md) | Annotate variants from text files using a variant (chromosome, position, ref, alt) key
`annotatevariants loci` | [**\[link\]**](AnnotateVariantsLoci.md) | Annotate variants from text files using a locus (chromosome, position) key
`annotatevariants intervals` | [**\[link\]**](AnnotateVariantsIntervals.md) | Annotate intervals of variants from text interval lists
`annotatevariants bed` | [**\[link\]**](AnnotateVariantsBed.md) | Annotate intervals of variants from UCSC BED files
`annotatevariants vcf` | [**\[link\]**](AnnotateVariantsVCF.md) | Annotate variants with VCF variant metadata
`annotatevariants vds` | [**\[link\]**](AnnotateVariantsVDS.md) | Annotate variants with annotations from another VDS
`annotatevariants expr` | [**\[link\]**](AnnotateVariantsExpr.md) | Annotate variants programmatically using the Hail expr language

#### Multiallelics, `annotatevariants`, and you

Three of the above commands involve joining two sets of variant keys: `table`, `vds`, and `vcf`.  This join can sometimes interact badly with multiallelic variants, because all alternate alleles are considered as part of the variant key.  For example:

 - The variant `22:140012:A:T,TTT` will not be annotated by `22:140012:A:T` or `22:140012:A:TTT`
 - The variant `22:140012:A:T` will not be annotated by `22:140012:A:T,TTT`

It is possible that an unsplit dataset contains no multiallelic variants, so ignore any warnings Hail prints if you know that to be the case.  Otherwise, run `splitmulti` before `annotatevariants` or use argument `--split` (for vcf/vds) if this is a concern.


