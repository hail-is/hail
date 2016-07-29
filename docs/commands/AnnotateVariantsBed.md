# `annotatevariants bed`

This module is a subcommand of `annotatevariants`, and annotates intervals of variants from UCSC BED files.

#### Command Line Arguments

Argument | Shortcut | Default | Description
:-:  | :-: |:-: | ---
`--input <file>` | `-i` | **Required** | Path to file
`--root <root>` | `-r` | **Required** | Annotation path root: period-delimited path starting with `va`

____

#### Description

UCSC bed files can have up to 12 fields, but Hail will only ever look at the first four.  The first three fields are required (`chrom`, `chromStart`, and `chromEnd`).  If a fourth column is found, Hail will parse this field as a string and load it into the specified annotation path.  If the bed file has only three columns, Hail will assign each variant a boolean annotation based on whether that variant was a member of any interval.

____

#### Examples

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