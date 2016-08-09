<div class="cmdhead"></div>

<div class="description"></div>

<div class="synopsis"></div>

<div class="options"></div>

<div class="cmdsubsection">
### Notes:
 
**Using the `--code` / `--root` arguments:**

Using this module requires one of these two arguments, and they are used to tell Hail how to insert the other vds annotations into the variant annotation schema.  The `--root` argument is the simpler of these two, and simply packages up all variant annotations from the second VDS as a `Struct` and drops it at the given `--root` location.  The `--code` argument expects an annotation expression just like [`annotatevariants expr`](AnnotateVariantsExpr.md), where you have access to `va` (the variant annotations in the VDS), and `vds`, the variant annotations in the input VDS.  `--root va.anno` is equivalent to `--code 'va.anno = vds'`.

**Here are some examples of common uses for the `--code` argument:**

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

**Multi-allelic variants**

This command involves joining two sets of variant keys, which can sometimes interact badly with multi-allelic variants, because all alternate alleles are considered as part of the variant key.  For example:

 - The variant `22:140012:A:T,TTT` will not be annotated by `22:140012:A:T` or `22:140012:A:TTT`
 - The variant `22:140012:A:T` will not be annotated by `22:140012:A:T,TTT`

It is possible that an unsplit dataset contains no multiallelic variants, so ignore any warnings Hail prints if you know that to be the case.  Otherwise, run `splitmulti` before `annotatevariants vds` or use argument `--split` if this is a concern.

</div>

<div class="cmdsubsection">
### Examples:

<h4 class="example">Annotate variants from a VDS with all fields</h4>
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

<h4 class="example">Annotate variants from a VDS using user-defined fields</h4>

If we don't want all the info field / qual / filter annotations, we can just pick specific ones using `--code`:

```
$ hail [read / import / previous commands] \
    annotatevariants vds \
        -i /user/me/myfile.vds \
        -c 'va.custom_annotation_1 = vds.custom_annotation_1'
```
</div>