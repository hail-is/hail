# Importing annotations into Hail

Hail includes modules for importing annotations from external files, for use in downstream filtering and analysis.  There are two such modules, one for annotating samples and one for annotating variants:
 - `annotatesamples`
 - `annotatevariants` [Skip to annotate variants section](#AnnoVar)
 
## Annotating Samples

Hail currently supports annotating samples from [tsv files](#SampleTSV) and [programmatic commands](#SampleProg).

____

<a name="SampleTSV"></a>
### Tab separated values (tsv[.gz])

This file format requires one column containing sample IDs, and each other column will be written to sample annotations.

**Command line arguments:**
- `-c <path-to-tsv>, --condition <path-to-tsv>` specify the file path **(Required)**
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
hail [read / import / previous commands] \
    annotatesamples \
        -c file:///user/me/samples.tsv \
        -t "Phenotype1: Double, Age: Int"
        -r sa.phenotypes
```
   
   This will read the tsv and produce annotations of the following schema:
   
```
Sample annotations:  sa: sa.<identifier>
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
hail [read / import, previous commands] \
    annotatesamples \
        -c file:///user/me/samples2.tsv \
        -s PT-ID
        --missing "."
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

There are currently five file types supported for annotating variants:

1. **Tab separated values (tsv, tsv.gz).**  This file format **requires** 4 columns for contig, position, ref, and alt.  Each other column in the file will be written to variant annotations.  The following command line arguments exist for .tsv files:
 - `-v | --vcolumns <columns>` -- specify the column headers for the contig, position, ref, and alt fields.  (Default: `Chromosome,Position,Ref,Alt`)
 - `-t | --types <type string>` -- specify data types of fields, in a comma-delimited string of `name:Type` elements.  If a field is not found in this type map, it will be interpreted as a string. (optional)
 - `-m | --missing <missing values>` -- specify identifiers to be treated as missing, in a comma-separated list.  (Default: `NA`)
2. **VCF (vcf, vcf.gz, vcf.bgz).**  This file format **requires** the `--root` command line option so that all info field annotations in the variant dataset are not overwritten.
4. **Interval list (.interval_list, .interval_list.gz).**  This file extension encompasses two file formats, `chr:start-end` and `chr start end strand target` (tsv).  The former will produce a boolean annotation, while the latter will store the `target` as a string.  The following argument is **required** for interval_list files:
 - `-i | --identifier <name>` -- Choose the name of the annotation in the vds.  If a `root` is specified, it can be found in `va.root.identifier`, otherwise `va.identifier`.
5. **UCSC bed (.bed, .bed.gz).**  This format is similar to the interval_list format.  The annotation name is designated in the track header of the bed file (`name="identifier"`).  If the body of the file contains the fourth (name) column, the annotation will be stored as a string with that field, otherwise boolean.  The spec for UCSC BED files is defined  [here.](https://genome.ucsc.edu/FAQ/FAQformat.html#format1)
6. **Hail-processed RDDs (.faf).**  Large TSV and VCF files can be very slow to parse and load into memory.  Since we still want to load these files, Hail supports reading pre-parsed and serialized files generated with the `convertannotations` module. [See conversion documentation here.](ConvertAnnotations.md)