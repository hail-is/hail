# Importing annotations into Hail

Hail includes modules for importing annotations from external files, for use in downstream filtering and analysis.  There are two such modules, one for annotating samples and one for annotating variants:
 - `annotatesamples`
 - `annotatevariants` [(click to skip to annotate variants section)](#AnnoVar)
 
Each of these modules has multiple run modes, which are specified with the first word after the command invocation:
```
<...> annotatesamples tsv <args>
```

```
<...> annotatevariants expr <args>
```

Read on and see examples for more information.
 
## Annotating Samples

Hail currently supports annotating samples from:
 - [tsv files](#SampleTSV)
 - [programmatic commands](#SampleProg).

____

<a name="SampleTSV"></a>
### Tab separated values (.tsv[.gz])

This file format requires one column containing sample IDs, and each other column will be written to sample annotations.

**Command line arguments:**

- `tsv` Invoke this functionality (`annotatesamples tsv <args>`)
- `-i <path-to-tsv>, --input <path-to-tsv>` specify the file path **(Required)**
- `-r <root>, --root <root>` specify the annotation path in which to place the fields read from the TSV, as a period-delimited path starting with `sa` **(Required)**
- `-s <id>, --sampleheader <id>` specify the name of the column containing sample IDs **(Optional with default "Sample")**
- `-t <typestring>, --types <typestring>` specify data types of fields, in a comma-delimited string of `name: Type` elements.  If a field is not found in this type map, it will be read and stored as a string **(Optional)** 
- `-m <missings>, --missing <missings>` specify identifiers to be treated as missing, in a comma-separated list **(Optional with default "NA")** 
 
____

**Example 1:**

```
$ cat ~/samples.tsv
Sample  Phenotype1   Phenotype2  Age
PT-1234 24.15        ADHD        24
PT-1235 31.01        ADHD        25
PT-1236 25.95        Control     19
PT-1237 26.80        Control     42
PT-1238 NA           ADHD        89
PT-1239 27.53        Control     55
```

To annotate from this file, one must a) specify where to put it in sample annotations, and b) specify the types of `Phenotype1` and `Age`.  The sample header agrees with the default ("Sample") and the missingness is encoded as "NA", also the default.  The Hail command line should look like:

```
$ hail [read / import / previous commands] \
    annotatesamples tsv \
        -i file:///user/me/samples.tsv \
        -t "Phenotype1: Double, Age: Int" \
        -r sa.phenotypes
```
   
   This will read the tsv and produce annotations of the following schema:
   
```
Sample annotations:  
sa: sa.<identifier>
    phenotypes: sa.phenotypes.<identifier>
        Phenotype1: Double
        Phenotype2: String
        Age: Int
```
   
____

**Example 2:**

```
$ cat ~/samples2.tsv
Batch   PT-ID
1kg     PT-0001
1kg     PT-0002
study1  PT-0003
study3  PT-0003
.       PT-0004
1kg     PT-0005
.       PT-0006
1kg     PT-0007
```

This file does not have non-string types, but it does have a sample column identifier that is not "Sample", and missingness encoded by "." instead of "NA".  The command line should read:

```
$ hail [read / import, previous commands] \
    annotatesamples tsv \
        -i file:///user/me/samples2.tsv \
        -s PT-ID \
        --missing "." \
        -r sa.group1.batch
```

____
   
<a name="SampleProg"></a>
### Programmatic Annotation

Programmatic annotation means computing new annotations from the existing exposed data structures, which in this case are the sample (`s`) and the sample annotations (`sa`).

**Command line arguments:**

 - `expr` Invoke this functionality (`annotatesamples expr <args>`)
 - `-c <condition>, --condition <condition>` 
 
 For more information, see [programmatic annotation documentation](ProgrammaticAnnotation.md)
 
____

<a name="AnnoVar"></a>
## Annotating Variants

Hail currently supports annotating variants from seven sources:
 - [TSV files](#VariantTSV)
 - [interval list files](#IntervalList)
 - [UCSC bed files](#UCSCBed)
 - [VCFs (info field, filters, qual)](#VCF)
 - [VDS (Hail internal representation)](#VDS)
 - [Programmatic commands](#VariantProg)

____

<a name="VariantTSV"></a>
### Tab separated values (tsv[.gz])

This file format requires either one column of the format "Chr:Pos:Ref:Alt", or four columns (one for each of these fields).  All other columns will be written to variant annotations.  Multiple files can be read in one command, but they must agree in their file format.

**Command line arguments:**

- `tsv` Invoke this functionality (`annotatevariants tsv <args>`)
- `<files...>` specify the file or files to be read **(Required)**
- `-r <root>, --root <root>` specify the annotation path in which to place the fields read from the TSV, as a period-delimited path starting with `va` **(Required)**
- `-v <variantcols>, --vcolumns <variantcols>` Either one column name (if Chr:Pos:Ref:Alt), or four comma-separated column identifiers **(Optional with default "Chromosome, Position, Ref, Alt")**
- `-t <typestring>, --types <typestring>` specify data types of fields, in a comma-delimited string of `name: Type` elements.  If a field is not found in this type map, it will be read and stored as a string **(Optional)** 
- `-m <missings>, --missing <missings>` specify identifiers to be treated as missing, in a comma-separated list **(Optional with default "NA")** 

____

**Example 1**
```
$ zcat ~/consequences.tsv.gz
Variant             Consequence     DNAseSensitivity
1:2001020:A:T       Synonymous      0.86
1:2014122:TTC:T     Frameshift      0.65
1:2015242:T:G       Missense        0.77
1:2061928:C:CCCC    Intergenic      0.12
1:2091230:G:C       Synonymous      0.66
```

This file contains one field to identify the variant and two data columns: one which encodes a string and one which encodes a double.  The command line should appear as:

```
$ hail [read / import / previous commands] \
    annotatevariants tsv \
        file:///user/me/consequences.tsv.gz \
        -t "DNAseSensitivity: Double" \
        -r va.varianteffects \
        -v Variant
```

This invocation will annotate variants with the following schema:

```
Variant annotations:  
va: va.<identifier>
    <probably lots of other stuff here>
    varianteffects: va.varianteffects.<identifier>
        Consequence: String
        DNAseSensitivity: Double
```

____

**Example 2**

```
$ zcat ~/ExAC_Counts.tsv.gz
Chr  Pos        Ref     Alt     AC
16   29501233   A       T       1
16   29561200   TTTTT   T       15023
16   29582880   G       C       10

```

In this case, the variant is indicated by four columns, but the header does not match the default ("Chromosome, Position, Ref, Alt").  The proper command line is below:

```
$ hail [read / import / previous commands] \
    annotatevariants tsv \
        file:///user/me/ExAC_Counts.tsv.gz \
        -t "AC: Int" \
        -r va.exac \
        -v "Chr,Pos,Ref,Alt"
```

And the schema:

```
Variant annotations:  
va: va.<identifier>
    <probably lots of other stuff here>
    exac: va.exac.<identifier>
        AC: Int
```

____

<a name="IntervalList"></a>
### Interval list files (.interval_list[.gz])

Interval list files annotate variants in certain regions of the genome.  Depending on the format of the file used, these can annotate variants with a boolean (whether the variant was in a window in the file) or a string (the target field).

**Command line arguments:**

- `intervals` Invoke this functionality (`annotatevariants intervals <args>`)
- `-i <path-to-interval-list>, --input <path-to-interval-list>` specify the file path **(Required)**
- `-r <root>, --root <root>` specify the annotation path in which to place the field read from the interval list, as a period-delimited path starting with `va` **(Required)**

There are two formats for interval list files.  The first appears as `chromosome:start-end` as below.  This format will annotate variants with a **boolean**, which is `true` if that variant is found in any interval specified in the file and `false` otherwise.
    
```
$ cat exons.interval_list
1:5122980-5123054
1:5531412-5531715
1:5600022-5601025
1:5610246-5610349
```

The second interval list format is a TSV with fields chromosome, start, end, strand, target.  **There should not be a header.**  This file will annotate variants with the **string** in the fifth column (target)

```
$ cat exons2.interval_list
1   5122980 5123054 + gene1
1   5531412 5531715 + gene1
1   5600022 5601025 - gene2
1   5610246 5610349 - gene2
```

____

**Example 1**

In this case, we will use as an example the first file above (`exons.interval_list`).  The command line should look like:

```
$ hail [read / import / previous commands] \
    annotatevariants intervals \
        -i file:///user/me/exons.interval_list \
        -r va.inExon \
```

This command will produce a schema like:

```
Variant annotations:  
va: va.<identifier>
    <probably lots of other stuff here>
    inExon: Boolean
```

____

**Example 2**

In this case, we will use the second file, `exons2.interval_list`.  This file contains a mapping between intervals and gene name.  The command looks similar:

```
$ hail [read / import / previous commands] \
    annotatevariants \
        -i file:///user/me/exons2.interval_list \
        -r va.gene \
```

This mode produces a string annotation:


```
Variant annotations:  
va: va.<identifier>
    <probably lots of other stuff here>
    gene: String
```

____

<a name="UCSCBed"></a>
### UCSC Bed files (.bed[.gz])


UCSC bed files [(see the website for spec)](https://genome.ucsc.edu/FAQ/FAQformat.html#format1) function similarly to .interval_list files, with a slightly different format.  Like interval list files, bed files can produce either a string or boolean annotation, depending on the presence of the 4th column (the target).  

**Command line arguments:**

- `bed` Invoke this functionality (`annotatevariants bed <args>`)
- `-i <path-to-bed>, --input <path-to-bed>` specify the file path **(Required)**
- `-r <root>, --root <root>` specify the annotation path in which to place the field read from the bed, as a period-delimited path starting with `va` **(Required)**

UCSC bed files can have up to 12 fields, but Hail will only ever look at the first four.  The first three fields are required (`chrom`, `chromStart`, and `chromEnd`).  If a fourth column is found, Hail will parse this field as a string and load it into the specified annotation path.  If the bed file has only three columns, Hail will assign each variant a boolean annotation based on whether that variant was a member of any interval.

____

**Example 1**

```
$ cat file1.bed
track name="BedTest"
20    1          14000000
20    17000000   18000000
```

In order to annotate with this file, the command should appear as:

```
$ hail [read / import / previous commands] \
    annotatevariants bed \
        -i file:///user/me/file1.bed \
        -r va.cnvRegion \
```

This file format produces a boolean annotation:

```
Variant annotations:  
va: va.<identifier>
    <probably lots of other stuff here>
    cnvRegion: Boolean
```

____

**Example 2**

```
$ cat file2.bed
browser position 20:1-18000000
browser hide all
track name="BedTest"
itemRgb="On"
20    1          14000000  cnv1
20    17000000   18000000  cnv2
```

This file has a more complicated header, but that does not affect Hail's parsing because the header is always skipped (Hail is not a genome browser).  However, it also has a fourth column, so this column will be parsed as a string.  The command line should follow the same format:

```
$ hail [read / import / previous commands] \
    annotatevariants bed \
        -i file:///user/me/file2.bed \
        -r va.cnvRegion \
```

The schema will reflect that this annotation is read as a string.

```
Variant annotations:  
va: va.<identifier>
    <probably lots of other stuff here>
    cnvRegion: String
```

____

<a name="VCF"></a>
### VCF files (.vcf[.bgz])

Hail can read vcf files to annotate a variant dataset.  Since Hail internally calls out to its `importvcf` module, these files must follow the same format / spec as described in the [importvcf module documentation](Importing.md).  Any VCF to spec will function in this module, but "info-only" VCFs are preferred (no samples, no format field in each line, 8 fields total).  Hail will read the same annotations as described in the importing documentation linked above.

**Command line arguments:**

- `vcf` Invoke this functionality (`annotatevariants vcf <args>`)
- `<files...>` specify the path to the file/files **(Required)**
- `-r <root>, --root <root>` specify the annotation path in which to place the annotations read from the vcf, as a period-delimited path starting with `va` **(Required)**

____

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
        /user/me/1kg.chr22.vcf.bgz \
        -r va.1kg \
```

The schema will include all the standard VCF annotations, as well as the info field:

```
Variant annotations:
va: va.<identifier>   
    <probably lots of other stuff here>
    1kg: va.1kg.<identifier>
        pass: Boolean
        filters: Set[String]
        rsid: String
        qual: Double
        cnvRegion: String
        info: va.1kg.info.<identifier>
            AC: Int
            AF: Double
            AN: Int
            BaseQRankSum: Double
```

<a name="VDS"></a>
### Hail variant dataset (.vds) files

Hail can also annotate from other VDS files.  This functionality calls out to the `read` module, and merges variant annotations into the current dataset.  These files may come from standard imported datasets, or could be generated by Hail's [preprocess annotations module](PreprocessAnnotations.md).

**Command line arguments:**

- `vds` Invoke this functionality (`annotatevariants vds <args>`)
- `-i <path-to-vds>, --input <path-to-vds>` specify the path to the VDS **(Required)**
- `-r <root>, --root <root>` specify the annotation path in which to place the annotations read from the vds, as a period-delimited path starting with `va` **(Required)**

____

**Example 1**

```
$ hail read /user/me/myfile.vds showannotations
hail: info: running: read /user/me/myfile.vds
hail: info: running: showannotations
Sample annotations:
sa: Empty

Variant annotations:
va: va.<identifier>
  rsid: String
  qual: Double
  filters: Set[String]
  pass: Boolean
  info: va.info.<identifier>
    AC: Int
  custom_annotation_1: Double
```

The above VDS file was imported from a VCF, and thus contains all the expected annotations from VCF files, as well as one user-added custom annotation (`va.custom_annotation_1`).  The proper command line to import it is below:

```
$ hail [read / import / previous commands] \
    annotatevariants vds \
        -i /user/me/myfile.vds \
        -r va.other \
```

The schema produced will look like this:

```
Variant annotations:
va: va.<identifier>   
    <probably lots of other stuff here>
    other: va.other.<identifier>
        rsid: String
        qual: Double
        filters: Set[String]
        pass: Boolean
        info: va.1kg.info.<identifier>
            AC: Int
        custom_annotation_1: Double
```

____
   
<a name="VariantProg"></a>
### Programmatic Annotation

Programmatic annotation means computing new annotations from the existing exposed data structures, which in this case are the variant (`v`) and the variant annotations (`va`).

**Command line arguments:**

- `expr` Invoke this functionality (`annotatevariants expr <args>`)
`-c <condition>, --condition <condition>` 

For more information, see [programmatic annotation documentation](ProgrammaticAnnotation.md)
