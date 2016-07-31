# Importing annotations from text files

Importing large files into Hail annotations can be slow, and this commands can be cumbersome to write on the command line (types may need to be specified, etc.).  Hail includes the module `importannotations` to (a) do the slow parsing step once if a file will be used to annotate many datasets, and (b) allow you to do standard VDS operations on these files.  This module produces a sites-only VDS.  There is currently one subcommand:
 - `importannotations table`
 
#### Command Line Arguments

Argument | Shortcut | Default | Description
:-:  | :-: |:-: | ---
`<files...>` | `--` | **Required** | Input files (supports globbing e.g. `files.chr*.tsv`), header read from first appearing file
`--input <file>` | `-i` | **Required** | Path to file
`--variant-expr` | `-e` | **Required** | Define how to get the variant out of the table using an [expression](../HailExpressionLanguage.md)
`--code <annotation expr>` | `-c` | **All annotations loaded** | Annotation [expression](../HailExpressionLanguage.md) using `va` (the existing vds annotations) and `table` (the annotations loaded from the table)
`--types <type-mapping>` | `-t` | All cols `String` | specify data types of fields, in a comma-delimited string of `name: Type` elements.  Untyped columns will be loaded as `String` (imputation off) or imputed from the first few lines of the file (imputation on)
`--missing <missing-value>` | `-m` | `NA` | Indicate the missing value in the table, if not "NA"
`--delimiter <regex>` | `-d` | `\t` | Indicate the field delimiter
`--comment <comment-char>` | `--` | **None** | Skip lines starting with the given pattern
`--no-header` | `--` | -- | Indicate that the file has no header.  Columns will instead be read as numbered, from `_0, _1, _2, ... _N`
`--impute` | `--` | -- | Turn on type imputation (works for Int, Double, Boolean, and Variant)

____

The generality of this module allows it to load delimited text files, json, or a mixture of the two.  

#### Using the `--variant-expr` argument

This argument tells Hail how to get a variant out of your table.  To do this, you are given the hail expr language, and each column in the table is exposed.  Since a `Variant` is not a simple string, using this is a little bit more complicated than using `--sample-expr` in `annotatesamples table`.  There are three ways to make a variant out of a table:
  
  1. Use a column of type `Variant`.  These columns look like `CHR:POS:REF:ALT` or `CHR:POS:REF:ALT1,ALT2,...ALTN`.  You can get a column of type `Variant` by explictly typing it using `--types` (`-t variantCol: Variant`), or by allowing Hail to do type imputation with `--impute`.  The `--variant-expr` argument here looks like `-e <variantCol>`.  
  2. Use a string column that takes the format designated above in an expr variant constructor.  This looks like `-e Variant(<stringCol>)` 
  3. Use the variant constructor that takes chromosome (`String`), position (`Int`), ref (`String`), and alt (`String`).  This looks something like `-e 'Variant(Chromosome, Position, Ref, Alt)'` assuming that these columns have the correct types.  _Note:_ type imputation will often assign chromosomes to `Int` type, so use `-t 'Chromosome: String'` to fix this.  
    
#### Using the `--code` argument

This optional argument allows you to subset or transform the annotations in the table.

Here are some examples of common uses:

Rename unnamed columns:
```
-c 'va.annot = table._1, va.score1 = table._2'
```

Load only specific annotations from the table:
```
-c 'va = select(table, toKeep1, toKeep2, toKeep3)'
```

**Example 1**
```
$ zcat ~/consequences.tsv.bgz
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
    importannotations table \
        consequences.tsv.bgz \
        -e Variant 
        --impute \
    write \
        -o /user/tpot/consequences.vds
```

This invocation will produce a VDS with the following schema:

```
Variant annotations:  
va: Struct {
    Variant: Variant
    Consequence: String
    DNAseSensitivity: Double
}
```

Notice that `Variant` will be added to variant annotations by default, duplicating it unnecessarily in the VDS.  In order to avoid this, you could add an argument to the `importannotations` command line: `--code 'va = drop(table, Variant)'`.  This will drop the 'Variant' field from the table `Struct`.

____

**Example 2**

```
$ zcat ~/ExAC_Counts.tsv.bgz
Chr  Pos        Ref     Alt     AC
16   29501233   A       T       1
16   29561200   TTTTT   T       15023
16   29582880   G       C       10

```

In this case, the variant is indicated by four columns instead of one, so we need to use the `Variant` expr constructor.  Additionally, since there is only one non-variant annotation, we may also want to let `va` be just that annotation (an integer referring to ExAC allele count).

```
$ hail \
    importannotations table \
        file:///user/tpot/ExAC_Counts.tsv.bgz \
        -t "AC: Int, Pos: Int" \
        -c 'va = table.AC' \
        -e "Variant(Chr,Pos,Ref,Alt)" \
    write \
        -o /user/tpot/ExAC_Counts.vds
```

The schema:

```
Variant annotations:  
va: Int
```
