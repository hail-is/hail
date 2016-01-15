# Exporting to TSVs in Hail

Hail has three export modules which write to TSVs:
 - `exportsamples`
 - `exportvariants`
 - `exportgenotypes`
 
These three export modules take a condition argument (`-c`) similar to [filtering](Filtering.md) expressions, with a similar namespace as well.  However, the expression is not parsed as a boolean, but rather a comma-delimited list of fields or expressions to print.  These fields will be printed in the order they appear in the expression in the header and on each line.

Command line arguments: 
 - `-c` -- export expression (see below) or .columns file
 - `-o` -- file path to which output should be written

## Export modules

1. `exportsamples` will print one line per sample in the VDS.  The accessible namespace includes:
   - `s` (sample)
   - `sa` (sample annotations)
2. `exportvariants` will print one line per variant in the VDS.  The accessible namespace includes:
   - `v` (variant)
   - `va` (variant annotations)
3. `exportgenotypes` will print one line per unique (variant, sample) in the VDS.   **WARNING: This is an operation with an output length of M x N.  Use it wisely if you value your gigabytes.** The accessible namespace includes:
   - `g` (genotype)
   - `s` (sample)
   - `sa` (sample annotations)
   - `v` (variant)
   - `va` (variant annotations). 
   
## Designating output with .columns files

Hail supports reading in a file ending with ".columns" to assign column names and expressions.  This file should contain one line per desired column.  Each line should contain two fields, separated by a tab: the header name in the first, the expression in the second.  Below are two examples of acceptable columns files:

```
$ cat exportVariants.columns
VARIANT	v
PASS	va.pass
FILTERS	va.filters
MISSINGNESS	1 - va.qc.callRate)
```

```
$ cat exportGenotypes.columns
SAMPLE	s
VARIANT	v
GQ	g.gq
DP	g.dp
ANNO1	va.MyAnnotations.anno1
ANNO2	va.MyAnnotations.anno2
```
 
## Designating output with an expression

Much like [filtering](Filtering.md) modules, exporting allows flexible expressions to be written on the command line.  While the filtering modules expect an expression that evaluates to true or false, export modules expect a comma-separated list of fields to print.  These fields should take the form `IDENTIFIER = <expression>`.  Below are examples of acceptable export expressions:


```
exportvariants -c 'VARIANT = v, PASS = va.pass, FILTERS = va.filters, MISSINGNESS = 1 - va.qc.callRate'
```

```
exportgenotypes -c 'SAMPLE=s,VARIANT=v,GQ=g.gq,DP=g.dp,ANNO1=va.MyAnnotations.anno1,ANNO2=va.MyAnnotations.anno2'
```

Note that the above two expressions will result in identical output to the example .columns files above.

It is also possible to export without identifiers, which will result in a file with no header.  In this case, the expressions should look like the examples below:
```
exportsamples -c 's.id, sa.qc.rTiTv'
```
```
exportvariants -c 'v,va.pass,va.qc.MAF'
```
```
exportgenotypes -c 'v,s.id,g.gq'
```

**Note:** if some fields have identifiers and some do not, an error will be thrown.  Either each field must be identified, or each field should include only an expression.

## Accessible fields of exposed classes

**Genotype:** `g`
 - `g                  String` -- String representation of a genotype: looks like `GT:AD:DP:PL`
 - `g.ad:          (Int, Int)` -- allelic depth for each allele
 - `g.dp:                 Int` -- the total number of informative reads
 - `g.isHomRef:       Boolean` -- true if this call is `0/0`
 - `g.isHet:          Boolean` -- true if this call is heterozygous
 - `g.isHomVar:       Boolean` -- true if this call is `1/1`
 - `g.isCalledNonRef: Boolean` -- true if either `g.isHet` or `g.isHomVar` is true
 - `g.isCalled:       Boolean` -- true if the genotype is `./.`
 - `g.isNotCalled:    Boolean` -- true if genotype is not `./.`
 - `g.gq:                 Int` -- the value of the lowest non-zero PL, or 0 if `./.`
 - `g.nNonRef:            Int` -- the number of called alternate alleles
 - `g.pAB:             Double` -- p-value for pulling the given allelic depth from a binomial distribution with mean 0.5
 - `g.pAB(theta):      Double` -- p-value for pulling the given allelic depth from a binomial distribution with mean `theta`

 
**Variant:** `v`
 - `v:                 String` -- String representation of a variant: looks like `CHROM_POS_REF_ALT`
 - `v.contig:          String` -- string representation of contig, exactly as imported
 - `v.start:              Int` -- SNP position or start of an indel
 - `v.ref:             String` -- reference allele
 - `v.alt:             String` -- Alternate allele
 - `v.wasSplit:       Boolean` -- true if this variant was originally multiallelic
 - `v.inParX:         Boolean` -- true if in pseudo-autosomal region on chromosome X
 - `v.inParY:         Boolean` -- true if in pseudo-autosomal region on chromosome Y
 - `v.isSNP:          Boolean` -- true if both `v.ref` and `v.alt` are single bases
 - `v.isMNP:          Boolean` -- true if `v.ref` and `v.alt` are the same (>1) length
 - `v.isIndel:        Boolean` -- true if `v.ref` and `v.alt` are not the same length
 - `v.isInsertion:    Boolean` -- true if `v.ref` is shorter than `v.alt`
 - `v.isDeletion:     Boolean` -- true if `v.ref` is longer than `v.alt`
 - `v.isComplex:      Boolean` -- true if `v` is not an indel, but `v.ref` and `v.alt` length do not match
 - `v.isTransition:   Boolean` -- true if the polymorphism is a purine-purine or pyrimidine-pyrimidine switch
 - `v.isTransversion: Boolean` -- true if the polymorphism is a purine-pyrimidine flip
 - `v.nMismatch:          Int` -- the total number of bases in `v.ref` and `v.alt` that do not match
 
**Sample:** `s`
 - `s.id:              String` The ID of this sample, as read at import-time 
 
**Variant Annotations:** `va`

There are no mandatory methods for annotation classes.  Annotations are generated by hail modules, like the qc modules with the `--store` option.  However, there is an exception -- when a VCF file is imported, certain fields will be saved in annotations:
 - `va.pass:          Boolean` -- true if the variant contains `PASS` in the filter field (false if `.` or other)
 - `va.filters:   Set[String]` -- set containing the list of filters applied to a variant.  Accessible using `va.filters.contains("VQSRTranche99.5...")`, for example
 - `va.rsid:           String` -- rsid of the variant, if it has one ("." otherwise)
 - `va.qual:           Double` -- the number in the qual field
 - `va.info.<field>:      Any` -- matches (with proper capitalization) any defined info field.  Data types match the definition in the vcf header, and if the `Number` is "A", "R", or "G", the result will be stored in an array.
 
If `variantqc --store` has been run:
 - `va.qc.<FIELD>:        Any` -- matches (with proper capitalization) variant qc fields.  [See list of available computed statistics here.](QC.md)

**Sample Annotations:** `sa`

There are no mandatory methods for annotation classes.  Annotations are generated by hail modules, like the qc modules with the `--store` option.  However, the following namespace will be available if `sampleqc --store` has been run:
 - `sa.qc.<FIELD>:        Any` -- matches (with proper capitalization) variant qc fields.  [See list of available computed statistics here.](QC.md)

## Exporting groups of annotations

We have added a convenience to allow export of groups of annotations as one field.  As an example, we will use the [QC metrics](QC.md) which could be in the variant annotations class.  Any field in this group can be exported easily by using its identifier: `va.qc.MAF`, for example.  However, if one wants to export **all** QC statistics, there is an easier way than writing an expression with ~30 semicolon-delimited references.
 - `va.qc:             String` -- writing this motif will print every field in `va.qc`, separated by semicolons.