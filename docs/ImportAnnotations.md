# Importing annotations into Hail

Hail includes modules for importing annotations from external files, for use in downstream filtering and analysis.  There are two such modules, one for annotating samples and one for annotating variants:
 - `annotatesamples`
 - `annotatevariants` [(click to skip to annotate variants section)](#AnnoVar)
 
## Annotating Samples

Hail currently supports annotating samples from:
 - [tsv files](#SampleTSV)
 - [programmatic commands](#SampleProg).

____

<a name="SampleTSV"></a>
### Tab separated values (.tsv[.gz])

This file format requires one column containing sample IDs, and each other column will be written to sample annotations.

**Command line arguments:**
- `-c <path-to-tsv>, --condition <path-to-tsv>` specify the file path **(Required)**
- `-r <root>, --root <root>` specify the annotation path in which to place the fields read from the TSV, as a period-delimited list starting with `sa` **(Required)**
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
    annotatesamples \
        -c file:///user/me/samples.tsv \
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
    annotatesamples \
        -c file:///user/me/samples2.tsv \
        -s PT-ID \
        --missing "." \
        -r sa.group1.batch
```

***
   
<a name="SampleProg"></a>
### Programmatic Annotation

Programmatic annotation means computing new annotations from the existing exposed data structures, which in this case are the sample (`s`) and the sample annotations (`sa`).

**Command line arguments:**
 - `-c <condition>, --condition <condition>` 
 
 For more information, see [programmatic annotation documentation](ProgrammaticAnnotation.md)
 
____

<a name="AnnoVar"></a>
## Annotating Variants

Hail currently supports annotating variants from seven sources:
 - [tsv files](#VariantTSV)
 - [interval list files](#IntervalList)
 - [UCSC bed files](#UCSCBed)
 - [VCFs (info field, filters, qual)](#VCF)
 - [VDS (Hail internal representation)](#VDS)
 - [Hail-generated Fast Annotation Format files](#FAF)
 - [Programmatic commands](#VariantProg)

____

<a name="VariantTSV"></a>
### Tab separated values (tsv[.gz])

This file format requires either one column of the format "Chr:Pos:Ref:Alt", or four columns (one for each of these fields).  All other columns will be written to variant annotations.

**Command line arguments:**
- `-c <path-to-tsv>, --condition <path-to-tsv>` specify the file path **(Required)**
- `-r <root>, --root <root>` specify the annotation path in which to place the fields read from the TSV, as a period-delimited list starting with `va` **(Required)**
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
    annotatevariants \
        -c file:///user/me/consequences.tsv.gz \
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
    annotatevariants \
        -c file:///user/me/ExAC_Counts.tsv.gz \
        -t "AC: Int" \
        -r va.exac \
        -v "Chr,Pos,Ref,Alt"
```

____

<a name="IntervalList"></a>
### Interval list files (.interval_list[.gz])

Interval list files annotate variants in certain regions of the genome.  Depending on the format of the file used, these can annotate variants with a boolean (whether the variant was in a window in the file) or a string (the target field).

**Command line arguments:**
- `-c <path-to-interval-list>, --condition <path-to-interval-list>` specify the file path **(Required)**
- `-r <root>, --root <root>` specify the annotation path in which to place the field read from the interval list, as a period-delimited list starting with `va` **(Required)**

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
    annotatevariants \
        -c file:///user/me/exons.interval_list \
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
        -c file:///user/me/exons2.interval_list \
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
- `-c <path-to-bed>, --condition <path-to-bed>` specify the file path **(Required)**
- `-r <root>, --root <root>` specify the annotation path in which to place the field read from the bed, as a period-delimited list starting with `va` **(Required)**

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
    annotatevariants \
        -c file:///user/me/file1.bed \
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
    annotatevariants \
        -c file:///user/me/file2.bed \
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

Hail can read vcf files to annotate a variant dataset.  Since Hail internally calls out to its `importvcf` module, these files must follow the same format / spec as described in the [importvcf module documentation].  Hail will read the same annotations as described
UCSC bed files [(see the website for spec)](https://genome.ucsc.edu/FAQ/FAQformat.html#format1) function similarly to .interval_list files, with a slightly different format.  Like interval list files, bed files can produce either a string or boolean annotation, depending on the presence of the 4th column (the target).  

**Command line arguments:**
- `-c <path-to-bed>, --condition <path-to-bed>` specify the file path **(Required)**
- `-r <root>, --root <root>` specify the annotation path in which to place the field read from the bed, as a period-delimited list starting with `va` **(Required)**

UCSC bed files can have up to 12 fields, but Hail will only ever look at the first four.  The first three fields are required (`chrom`, `chromStart`, and `chromEnd`).  If a fourth column is found, Hail will parse this field as a string and load it into the specified annotation path.  If the bed file has only three columns, Hail will assign each variant a boolean annotation based on whether that variant was a member of any interval.


<a name="VDS"></a>
<a name="FAF"></a>
1. **Tab separated values (.tsv[.gz]).**  This file format **requires** 4 columns for contig, position, ref, and alt.  Each other column in the file will be written to variant annotations.  The following command line arguments exist for .tsv files:
 - `-v | --vcolumns <columns>` -- specify the column headers for the contig, position, ref, and alt fields.  (Default: `Chromosome,Position,Ref,Alt`)
 - `-t | --types <type string>` -- specify data types of fields, in a comma-delimited string of `name:Type` elements.  If a field is not found in this type map, it will be interpreted as a string. (optional)
 - `-m | --missing <missing values>` -- specify identifiers to be treated as missing, in a comma-separated list.  (Default: `NA`)
2. **VCF (vcf, vcf.gz, vcf.bgz).**  This file format **requires** the `--root` command line option so that all info field annotations in the variant dataset are not overwritten.
4. **Interval list (.interval_list, .interval_list.gz).**  This file extension encompasses two file formats, `chr:start-end` and `chr start end strand target` (tsv).  The former will produce a boolean annotation, while the latter will store the `target` as a string.  The following argument is **required** for interval_list files:
 - `-i | --identifier <name>` -- Choose the name of the annotation in the vds.  If a `root` is specified, it can be found in `va.root.identifier`, otherwise `va.identifier`.
5. **UCSC bed (.bed, .bed.gz).**  This format is similar to the interval_list format.  The annotation name is designated in the track header of the bed file (`name="identifier"`).  If the body of the file contains the fourth (name) column, the annotation will be stored as a string with that field, otherwise boolean.  The spec for UCSC BED files is defined  [here.](https://genome.ucsc.edu/FAQ/FAQformat.html#format1)
6. **Hail-processed RDDs (.faf).**  Large TSV and VCF files can be very slow to parse and load into memory.  Since we still want to load these files, Hail supports reading pre-parsed and serialized files generated with the `convertannotations` module. [See conversion documentation here.](ConvertAnnotations.md)