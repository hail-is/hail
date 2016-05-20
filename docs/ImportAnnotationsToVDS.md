# Importing annotations TSVs

Importing large files into Hail annotations can be slow, and these commands can be cumbersome to write on the command line (types may need to be specified, etc.).  Hail includes the module `importannotations` to do the slow parsing steps ahead of time, which is useful if a large will be used to annotate many datasets or if one wants to perform filtering and querying operations on these files.  Files are imported as a sites-only VDS.  There are two subcommands:
 - `importannotations tsv`
 - `importannotations json`

### Text tables

This subcommand expects text files with multiple delimited columns (default: tab-delimited).  Variants are keyed either by one column of the format "Chr:Pos:Ref:Alt", or four columns (one for each of these fields).  All other columns will be written to variant annotations as a struct.  Multiple files can be read in one command, but they must agree in their file format.

**Command line arguments:**
- `table` Invoke this functionality (`importannotations table <args>`)
- `<files>` specify the file or files to read **(Required)**
 - `-v <variantcols>, --vcolumns <variantcols>` Either one column identifier (if Chr:Pos:Ref:Alt), or four comma-separated column identifiers **(Optional with default "Chromosome, Position, Ref, Alt")**
 - `-t <typestring>, --types <typestring>` specify data types of fields, in a comma-delimited string of `name: Type` elements.  If a field is not found in this type map, it will be read and stored as a string **(Optional)** 
 - `-m <missings>, --missing <missings>` specify identifiers to be treated as missing, in a comma-separated list **(Optional with default "NA")** 

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
$ hail \
    importannotations \
        file:///user/tpot/consequences.tsv.gz \
        -t "DNAseSensitivity: Double" \
        -r va.varianteffects \
        -v Variant \
    write \
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
$ hail \
    importannotations \
        file:///user/tpot/ExAC_Counts.tsv.gz \
        -t "AC: Int" \
        -r va.exac \
        -v "Chr,Pos,Ref,Alt" \
    write \
        -o /user/tpot/ExAC_Counts.vds
```

### JSON

This `json` subcommand imports annotations from JSON files, one JSON object per line.  Variants are keyed by four expressions computing the chromosome (String), position (Int), ref (String) and alts (Array[String]) per JSON object.  The entire JSON object is written to the variant annotations.  Multiple files can be read in one command, but they must agree in their file format.

**Command line arguments:**

- `json` Invoke this functionality (`importannotations json <args>`)
- `<files...>` specify the file or files to be read **(Required)**
- `-v | --vfields <variantcols>` Four comma-delimited expressions computing the chromosome (String), position (Int), ref (String) and alts (Array[String]) **(Required)**
- `-t | --type <typestring>` type of the JSON objects  **(Required)**
