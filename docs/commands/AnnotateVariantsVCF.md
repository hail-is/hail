# `annotatevariants vcf`

This module is a subcommand of `annotatevariants`, and annotates variants from a VCF.

#### Command Line Arguments

Argument | Shortcut | Default | Description
:-:  | :-: |:-: | ---
`<files...>` |  | **Required** | Input files (supports globbing e.g. `files.chr*.vcf.bgz`), header read from first appearing file
`--root <root>` | `-r` | **This or `--code` required** | Annotation path: period-delimited path starting with `va`
`--code <annotation expr>` | `-c` | **This or `--root` required** | Annotation [expression](../HailExpressionLanguage.md) using `va` (the existing vds annotations) and `vcf` (the annotations loaded from the vds)
`--split` | | | Split multiallelic variants in the VCF before annotating
____

#### Using the `--code` / `--root` arguments

Using this module requires one of these two arguments, and they are used to tell Hail how to insert the table into the variant annotation schema.  The `--root` argument is the simpler of these two, and simply packages up all table annotations as a `Struct` and drops it at the given `--root` location.  The `--code` argument expects an annotation expression just like [`annotatevariants expr`](AnnotateVariantsExpr.md), where you have access to `va` (the variant annotations in the VDS), and `vcf`, a struct with all the standard vcf annotations.  `--root va.anno` is equivalent to `--code 'va.anno = vcf'`.

An example of a common use for the `--code` argument:

Overwrite info field from another vcf:
```
-c 'va.info = vcf.info'
```

____

#### Examples

**Example 1**

```
$ hdfs dfs -zcat 1kg.chr22.vcf.bgz
##fileformat=VCFv4.1
##FILTER =<ID=LowQual,Description="Low quality">
##INFO=<ID=AC,Number=A,Type=Integer,Description="Allele count in genotypes, for each ALT allele, in the same order as listed">
##INFO=<ID=AF,Number=A,Type=Float,Description="Allele Frequency, for each ALT allele, in the same order as listed">
##INFO=<ID=AN,Number=1,Type=Integer,Description="Total number of alleles in called genotypes">
##INFO=<ID=BaseQRankSum,Number=1,Type=Float,Description="Z-score from Wilcoxon rank sum test of Alt Vs. Ref base qualities">
 ...truncated...
#CHROM  POS             ID      REF     ALT     QUAL            FILTER  INFO
22      16050036        .       A       C       19961.13        .       AC=1124;AF=0.597;AN=1882;BaseQRankSum=2.875
22      16050115        .       G       A       134.13          .       AC=20;AF=7.252e-03;AN=2758;BaseQRankSum=5.043
22      16050159        .       C       T       12499.96        .       AC=689;AF=0.266;AN=2592;BaseQRankSum=-6.983
22      16050213        .       C       T       216.35          .       AC=25;AF=8.096e-03;AN=3088;BaseQRankSum=-2.275
22      16050252        .       A       T       22211           .       AC=1094;AF=0.291;AN=3754;BaseQRankSum=-7.052
22      16050408        .       T       C       83.90           .       AC=75;AF=0.015;AN=5026;BaseQRankSum=-18.144
 ...truncated...
```

The proper command line:

```
$ hail [read / import / previous commands] \
    annotatevariants vcf \
        /user/me/1kg.chr*.vcf.bgz \
        -r va.`1kg` \
```

The schema will include all the standard VCF annotations, as well as the info field:

```
Variant annotations:
va: Struct {
    <probably lots of other stuff here>
    `1kg`: Struct {
        pass: Boolean
        filters: Set[String]
        rsid: String
        qual: Double
        cnvRegion: String
        info: Struct {
            AC: Int
            AF: Double
            AN: Int
            BaseQRankSum: Double
        }
    }
}
```