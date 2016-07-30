# `annotatevariants table`

This module is a subcommand of `annotatevariants`, and loads a text file into variant annotations using a variant key or chr/pos/ref/alt keys.

#### Command Line Arguments

Argument | Shortcut | Default | Description
:-:  | :-: |:-: | ---
`<files...>` | `--` | **Required** | Input files (supports globbing e.g. `files.chr*.tsv`), header read from first appearing file
`--input <file>` | `-i` | **Required** | Path to file
`--variant-expr` | `-e` | **Required** | Define how to get the variant out of the table using an [expression](../HailExpressionLanguage.md)
`--root <root>` | `-r` | **This or `--code` required** | Annotation path: period-delimited path starting with `va`
`--code <annotation expr>` | `-c` | **This or `--root` required** | Annotation [expression](../HailExpressionLanguage.md) using `va` (the existing vds annotations) and `table` (the annotations loaded from the table)
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
    
#### Using the `--code` / `--root` arguments

Using this module requires one of these two arguments, and they are used to tell Hail how to insert the table into the variant annotation schema.  The `--root` argument is the simpler of these two, and simply packages up all table annotations as a `Struct` and drops it at the given `--root` location.  If your table had columns "Variant", "Consequence", and "Gene", then `--root va.metadata` would create the struct `{Variant, Consequence, Gene}` at `sa.metadata`, which would give you access to the paths `va.metadata.Variant`, `va.metadata.Consequence`, `va.metadata.Gene`.  The `--code` argument expects an annotation expression just like [`annotatevariants expr`](AnnotateVariantsExpr.md), where you have access to `va` (the variant annotations in the VDS), and `table`, a struct with all the columns in the table.  `--root va.anno` is equivalent to `--code 'va.anno = table'`.

Here are some examples of common uses for the `--code` argument:

Table with only one annotation column -- don't generate a full struct:
```
-c 'va.annot = table._1'
```

Want to put annotations on the top level under `va`:
```
-c 'va = merge(va, table)'
```

Want to load only specific annotations from the table:
```
-c 'va.annotations = select(table, toKeep1, toKeep2, toKeep3)'
```

The above is equivalent to:
```
-c 'va.annotations.toKeep1 = table.toKeep1, 
    va.annotations.toKeep2 = table.toKeep2,
    va.annotations.toKeep3 = table.toKeep3'
```

____

#### Examples

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

This file contains one field to identify the variant and two data columns: one which encodes a string and one which encodes a double.  Using the `--impute` option, we can avoid having to specify the `Variant` and `Double` types.  The command line should appear as:

```
$ hail [read / import / previous commands] \
    annotatevariants table \
        file:///user/me/consequences.tsv.gz \
        -r va.varianteffects \
        -e Variant
        --impute
```

This invocation will annotate variants with the following schema:

```
Variant annotations:
va: Struct {
    <probably lots of other stuff here>
    varianteffects: Struct {
        Consequence: String
        DNAseSensitivity: Double
    }
}
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
    annotatevariants table \
        file:///user/me/ExAC_Counts.tsv.gz \
        -t "AC: Int, Pos: Int" \
        -r va.exac \
        -e "Variant(Chr,Pos,Ref,Alt)"
```

And the schema:

```
Variant annotations:
va: Struct {
    <probably lots of other stuff here>
    exac: Struct {
        AC: Int
    }
}
```
