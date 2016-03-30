# Preprocessing annotations TSVs

Importing large TSV files into Hail annotations can be slow, and these commands can be cumbersome to write on the command line (types may need to be specified, etc.).  Hail includes the module `preprocessannotations` to do the slow parsing steps ahead of time, which is useful if a large TSV will be used to annotate many datasets.  This module reads in a TSV file, and writes a VDS file with no samples in which the TSV fields are stored as annotations.

Appropriate TSV files contain variant identifiers either in one column of the format "Chr:Pos:Ref:Alt", or four columns (one for each of these fields).  All other columns will be written to variant annotations.

**Command line arguments:**
 - `-c <path-to-tsv>, --condition <path-to-tsv>` specify the file path **(Required)**
 - `-v <variantcols>, --vcolumns <variantcols>` Either one column identifier (if Chr:Pos:Ref:Alt), or four comma-separated column identifiers **(Optional with default "Chromosome, Position, Ref, Alt")**
 - `-t <typestring>, --types <typestring>` specify data types of fields, in a comma-delimited string of `name: Type` elements.  If a field is not found in this type map, it will be read and stored as a string **(Optional)** 
 - `-m <missings>, --missing <missings>` specify identifiers to be treated as missing, in a comma-separated list **(Optional with default "NA")** 
 - `-o <output-file.vds>, --output <output-file.vds>` specify path to which the processed VDS should be written **(Required)**

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
$ hail preprocessannotations \
        -c file:///user/tpot/consequences.tsv.gz \
        -t "DNAseSensitivity: Double" \
        -r va.varianteffects \
        -v Variant
        -o /user/tpot/consequences.vds
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
$ hail preprocessannotations \
        -c file:///user/tpot/ExAC_Counts.tsv.gz \
        -t "AC: Int" \
        -r va.exac \
        -v "Chr,Pos,Ref,Alt"
        -o /user/tpot/ExAC_Counts.vds
```
