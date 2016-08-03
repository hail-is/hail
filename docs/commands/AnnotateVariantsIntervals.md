# `annotatevariants intervals`

This module is a subcommand of `annotatevariants`, and annotates intervals of variants from text interval lists.

#### Command Line Arguments

Argument | Shortcut | Default | Description
:-:  | :-: |:-: | ---
`--all` | `-a` | false | Annotate all matching values as Set[String]
`--input <file>` | `-i` | **Required** | Path to file
`--root <root>` | `-r` | **Required** | Annotation path root: period-delimited path starting with `va`

____

#### Description

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

If the `-a/--all` option is given, the annotation will be the set
(possibly empty) of fifth column strings (targets) as a `Set[String]`
for all intervals that overlap the given variant.

____

#### Examples

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
