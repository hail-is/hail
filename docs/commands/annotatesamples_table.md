<div class="cmdhead"></div>

<div class="description"></div>

<div class="synopsis"></div>

<div class="options"></div>

<div class="cmdsubsection">
### Notes

The generality of this module allows it to load delimited text files, json, or a mixture of the two.  

**Using the `--sample-expr` argument:**

This argument tells Hail how to get a sample ID out of your table.  To do this, you are given the hail expr language, and each column in the table is exposed.  This could be something like `-e Sample` (if your sample id is in a column called 'Sample'), `-e _2` (if your sample ID is the 3rd column of a table with no header), or it could be something more complicated like `-e 'if ("PGC" ~ ID1) ID1 else ID2'`.  All that matters is that this expr results in a `String`.  Note also that if the expr evaluates to `missing`, it will not be mapped to any VDS samples.

**Using the `--code` / `--root` arguments:**

Using this module requires one of these two arguments, and they are used to tell Hail how to insert the table into the sample annotation schema.  The `--root` argument is the simpler of these two, and simply packages up all table annotations as a `Struct` and drops it at the given `--root` location.  If your table had columns "Sample", "Sex", and "Batch", then `--root sa.metadata` would create the struct `{Sample, Sex, Batch}` at `sa.metadata`, which would give you access to the paths `sa.metadata.Sample`, `sa.metadata.Sex`, `sa.metadata.Batch`.  The `--code` argument expects an annotation expression just like [`annotatesamples expr`](AnnotateSamplesExpr.md), where you have access to `sa` (the sample annotations in the VDS), and `table`, a struct with all the columns in the table.  `--root sa.anno` is equivalent to `--code 'sa.anno = table'`.

**Here are some examples of common uses for the `--code` argument:**

Table with only one annotation column -- don't generate a full struct:
```
-c 'sa.annot = table._1'
```

Want to put annotations on the top level under `sa`:
```
-c 'sa = merge(sa, table)'
```

Want to load only specific annotations from the table:
```
-c 'sa.annotations = select(table, toKeep1, toKeep2, toKeep3)'
```

The above is equivalent to:
```
-c 'sa.annotations.toKeep1 = table.toKeep1, 
    sa.annotations.toKeep2 = table.toKeep2,
    sa.annotations.toKeep3 = table.toKeep3'
```
</div>

<div class="cmdsubsection">
### Examples

<h4 class="example">Import annotations from a tsv file with phenotypes and age</h4>

We have a file with phenotypes and age:
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

The appropriate command line to load this file into Hail:

```
$ hail [read / import / previous commands] \
    annotatesamples table \
        -e Sample
        -i file:///user/me/samples.tsv \
        -t "Phenotype1: Double, Phenotype2: Double, Age: Int" \
        -r sa.phenotypes
```

   This will read the file and produce annotations of the following schema:

```
Sample annotations:
sa: sa.<identifier>
    phenotypes: sa.phenotypes.<identifier>
        Phenotype1: Double
        Phenotype2: String
        Age: Int
```

<h4 class="example">Import annotations from a csv file with missing data and special characters</h4>

```
$ cat ~/samples2.tsv
Batch,PT-ID
1kg,PT-0001
1kg,PT-0002
study1,PT-0003
study3,PT-0003
.,PT-0004
1kg,PT-0005
.,PT-0006
1kg,PT-0007
```

In this case, we should do a few things differently:

 - 'escape' the PT-ID column with backticks in our `--sample-expr` argument because it contains a dash
 - pass the non-default delimiter ","
 - pass a non-default missing value "."  
 - Since this table only has one column useful to us, we can simply add that using an annotation expr argument (`--code`) instead of the `--root` argument. 
```
$ hail [read / import, previous commands] \
    annotatesamples table \
        -i file:///user/me/samples2.tsv \
        --delimiter ',' \
        --sample-expr '`PT-ID`' \
        --missing "." \
        --code 'sa.batch = table.Batch'
```

<h4 class="example">Import annotations from a file with no header and the sample IDs need to be transformed</h4>

In this example, let's suppose that our vds sample IDs are of the form `NA#####`.  This file has no header line, and the sample id is hidden in a field with other information.

```
$ cat ~/samples3.tsv
1kg_NA12345     female   
1kg_NA12346     male
1kg_NA12348     female
pgc_NA23456     female
pgc_NA23415     male
pgc_NA23418     male
```

Let's import it:

```
$ hail  [ previous commands ] \ 
    annotatesamples table \
        --no-header
        --sample-expr '_0.split("_")[1]'
        --code 'sa.sex = table._1, sa.batch = table._0.split("_")[0]'
```

Note that both the `--sample-expr` and `--code` arguments take greater advantage of expr language features.
</div>