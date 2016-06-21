# Programmatic annotation in Hail

Hail supports programmatic annotation, which is the computation of new annotations from the data structures and functions exposed in the Hail command-line language.  This can be as simple as creating an annotation `va.missingness` = `1 - va.qc.callRate`, or it can be a complex computation involving conditionals and multiple stored annotations.

**Command line arguments:**
 - `-c <condition>, --condition <condition>` annotation statement(s), see below

**Usage:**

```
    $ hail <read/import> annotatesamples expr -c 'expression'
```

```
    $ hail <read/import> annotatevariants expr -c 'expression'
```

```
    $ hail <read/import> annotateglobal -c 'expression'
```

## Namespace

**`annotatevariants`:**

Identifier | Description
:-: | ---
`v` | Variant
`va` | Variant annotations
`global` | global annotation
`gs` | Genotype row [aggregable](HailExpressionLanguage.md#aggregables)

**`annotatesamples`:**

Identifier | Description
:-: | ---
`s` | Sample
`sa` | Sample annotations
`global` | global annotation
`gs` | Genotype column [aggregable](HailExpressionLanguage.md#aggregables)

**`annotateglobal`:**

Identifier | Description
:-: | ---
`global` | Existing global annotations
`variants` | variants and their annotations, an [aggregable](HailExpressionLanguage.md#aggregables)
`samples` | samples and their annotations, an [aggregable](HailExpressionLanguage.md#aggregables)


## Command syntax
 
In both `annotatesamples` and `annotatevariants`,  the syntax for creating programmatic annotations is the same.  These commands are composed of one or more annotation statements of the following structure:

```
annotatevariants expr \
   -c   "va.path.to.new.annotation          =         if (va.pass) va.qc.gqMean else 0    [ , ... ]"
                  ^                         ^                     ^                           ^
         *period-delimited path         equals sign            expression               comma followed by
       starting with 'va' or 'sa'     delimits path                                     more annotation
       where the annotation will      and expression                                       statements
             be placed
```

In this example, `annotatevariants` will produce a new annotation stored in `va.path.to.new.annotation`, which is accessible in all future modules. 

<sup>*</sup> Note: if a field exists in annotations that contains characters 
not valid in simple identifiers (like whitespace, `$`, `%`, etc...), 
access it by escaping an identifier with backticks: 
```
va.custom.`this field has whitespace`

sa.`identifier...with...dots`.annotation1

va.`layer one`.`layer two`.`layer three`
```

____

## Example commands
 
```
hail read -i my.vds \
 sampleqc \
 annotatesamples expr -c 'sa.missingness = 1 - sa.qc.callRate'
```

```
hail read -i my.vds \
 sampleqc \
 annotatesamples expr -c 'sa.myannotations.indelSnpRatio = (sa.qc.nInsertion + sa.qc.nDeletion) / sa.qc.nSNP.toDouble'
```

```
hail importvcf my.vcf.bgz \
 annotatevariants expr -c 'va.custom.percentCalled = 100 * sa.qc.callRate'
```

```
hail read -i my.vds \
 variantqc \
 annotatevariants expr -c 'va.a = 2, va.b = 3, va.c = 4, va.d = if (va.pass) va.qc.pHWE else 1'
```

```
hail read -i my.vds \
 variantqc \
 annotatevariants expr -c 'va.a = 2, va.b = 3, va.c = 4, va.d = if (va.pass) va.qc.pHWE else 1'
```

```
hail read -i my.vds \
 variantqc \
 annotatevariants expr -c 'va.info.AC = va.qc.AC'
```

____

 
## Order of operations

When you generate more than one new annotation, the operations are structured in the following order:

1.  All computations are evaluated from the annotation state at the beginning of the current module.  This means that you cannot write `annotatevariants -c 'va.a = 5, va.b = va.a * 2'` -- `va.a` does not exist in the base schema, so it cannot be found during the computations.
2.  Each statement generates a new schema based on the schema to its left (or the base, for the first command).  These transformations overwrite previous annotations in their path, i.e. if your dataset contains `va.a: Int` and you insert `va.a.b = 10`, `va.a: Int` is overwritten.  These overwrites happen left to right: 

```
annotatevariants expr -c 'va.a = 1, va.a = 2, va.a = 3'
```

The above command will produce one annotation, where `va.a` is `3`.  You should not do this because it is silly.

## Reasons to use this functionality

Some common reasons to create new annotations from existing ones are:

1. **Simplifying a workflow.**  If there is a metric computed from other annotations that is used in multiple places in a workflow, it should be placed in annotations.  This will also have performance benefits if the metric is expensive to compute.
2. **VCF info field annotations.**  Hail does not recompute the metrics in `va.info` when samples or genotypes are filtered, so something like `info.AC` does not necessarily reflect the state of data passed to `exportvcf`.  This can be resolved by copying annotations to `va.info`.  [See additional information here.](ExportVCF.md#annotations)
