<div class="cmdhead"></div>

<div class="description"></div>

<div class="synopsis"></div>

<div class="options"></div>

<div class="subsection">
### Notes:

[UCSC bed files](https://genome.ucsc.edu/FAQ/FAQformat.html#format1) can have up to 12 fields, but Hail will only ever look at the first four.  The first three fields are required (`chrom`, `chromStart`, and `chromEnd`).  If a fourth column is found, Hail will parse this field as a string and load it into the specified annotation path.  If the bed file has only three columns, Hail will assign each variant a boolean annotation based on whether that variant was a member of any interval.

If the `-a/--all` option is given and a fourth column is present, the
annotation will be the set (possibly empty) of fourth column strings
as a `Set[String]` for all intervals that overlap the given variant.

**NOTE:** UCSC BED files are 0-indexed, which means that the line "5  100  105" will include the loci `5:99, 5:100, 5:101, 5:102, 5:103`.  The last locus included in this interval is two smaller than the listed end!

</div>

<div class="subsection">
### Examples:

<h4 class="example">Importing data from a standard BED file</h4>

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


<h4 class="example">Importing data from a bed file with extra header information</h4>

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
