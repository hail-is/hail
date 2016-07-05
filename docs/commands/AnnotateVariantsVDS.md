# `annotatevariants vds`

This module is a subcommand of `annotatevariants`, and annotates variants from another VDS.

#### Command Line Arguments

Argument | Shortcut | Default | Description
:-:  | :-: |:-: | ---
`--input <vds-path>` | `-i` | **Required** | Path to VDS
`--root <root>` | `-r` | **This or `--code` required** | Annotation path: period-delimited path starting with `va`
`--code <annotation expr>` | `-c` | **This or `--root` required** | Annotation [expression](../HailExpressionLanguage.md) using `va` (the existing vds annotations) and `table` (the annotations loaded from the table)
`--split` | | | Split multiallelic variants in the input VDS before annotating

____

#### Using the `--code` / `--root` arguments

Using this module requires one of these two arguments, and they are used to tell Hail how to insert the other vds annotations into the variant annotation schema.  The `--root` argument is the simpler of these two, and simply packages up all variant annotations from the second VDS as a `Struct` and drops it at the given `--root` location.  The `--code` argument expects an annotation expression just like [`annotatevariants expr`](AnnotateVariantsExpr.md), where you have access to `va` (the variant annotations in the VDS), and `vds`, the variant annotations in the input VDS.  `--root va.anno` is equivalent to `--code 'va.anno = vds'`.

Here are some examples of common uses for the `--code` argument:

Select only one VDS annotation:
```
-c 'va.annot = vds.anno1'
```

Want to put annotations on the top level under `va`:
```
-c 'va = merge(va, vds)'
```

Want to load only specific annotations from the table:
```
-c 'va.annotations = select(vds, toKeep1, toKeep2, toKeep3)'
```

The above is equivalent to:
```
-c 'va.annotations.toKeep1 = vds.toKeep1, 
    va.annotations.toKeep2 = vds.toKeep2,
    va.annotations.toKeep3 = vds.toKeep3'
```

____

#### Examples

____

The below VDS file was imported from a VCF, and thus contains all the expected annotations from VCF files, as well as one user-added custom annotation (`va.custom_annotation_1`).

```
$ hail read /user/me/myfile.vds printschema
hail: info: running: read /user/me/myfile.vds
hail: info: running: printschema
Sample annotations:
sa: Empty

Variant annotations:
va: Struct {
  rsid: String
  qual: Double
  filters: Set[String]
  pass: Boolean
  info: Struct {
    AC: Int
  }
  custom_annotation_1: Double
}
```

If we want to annotate another VDS from this one, one proper command line to import it is below:

```
$ hail [read / import / previous commands] \
    annotatevariants vds \
        -i /user/me/myfile.vds \
        -r va.other
```

The schema produced will look like this:

```
Variant annotations:
va: Struct {
    <probably lots of other stuff here>
    other: Struct {
        rsid: String
        qual: Double
        filters: Set[String]
        pass: Boolean
        info: Struct {
            AC: Int
        }
        custom_annotation_1: Double
    }
}
```

If we don't want all the info field / qual / filter annotations, we can just pick specific ones using `--code`:

```
$ hail [read / import / previous commands] \
    annotatevariants vds \
        -i /user/me/myfile.vds \
        -c 'va.custom_annotation_1 = vds.custom_annotation_1'
```
