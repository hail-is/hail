# Programmatic annotation in Hail

Hail supports programmatic annotation, which is the computation of new annotations from the data structures and functions exposed in the Hail command-line language.  This can be as simple as creating an annotation `va.missingness` = `1 - va.qc.callRate`, or it can be a complex computation involving conditionals and multiple stored annotations.

## Example commands

```
hail read -i my.vds \
    sampleqc \
    annotatesamples -c 'sa.missingness = 1 - sa.qc.callRate'
```

```
hail read -i my.vds \
    sampleqc \
    annotatesamples -c 'sa.myannotations.indelSnpRatio = (sa.qc.nInsertion + sa.qc.nDeletion) / sa.qc.nSNP.toDouble'
```

```
hail importvcf my.vcf.bgz \
    annotatevariants -c 'va.custom.percentCalled = 100 * sa.qc.callRate'
```

```
hail read -i my.vds \
    variantqc \
    annotatevariants -c 'va.a = 2, va.b = 3, va.c = 4, va.d = if (va.pass) va.qc.pHWE else 1'
```


## Command syntax
 
In both `annotatesamples` and `annotatevariants`,  the syntax for creating programmatic annotations is the same.  These commands are composed of one or more annotation statements of the following structure:

```
annotatevariants -c   "va.path.to.new.annotation          =         if (va.pass) va.qc.gqMean else 0    [ , ... ]"
                                ^                         ^                     ^                           ^
                       *period-delimited path         equals sign            expression               comma followed by
                     starting with 'va' or 'sa'     delimits path                                     more annotation
                                                    and expression                                       statements
```                                                 
`*` Note: if a field exists in annotations that contains characters 
not valid in simple identifiers (like whitespace, $, %, etc...), 
access it with backticks using this format: 
```
va.custom.`this field has whitespace`
sa.`ExcitingAnnotations!!!!!`.annotation1
```
 
In this example, `annotatevariants` will produce a new annotation stored in `va.path.to.new.annotation`, which is accessible in all future modules. 
 
## Order of operations

When you generate more than one new annotation, the operations are structured in the following order:

1.  All computations are evaluated from the annotation state at the beginning of the current module.  This means that you cannot write `annotatevariants -c 'va.a = 5, va.b = va.a * 2'` -- `va.a` does not exist in the base schema, so it cannot be found during the computations.
2.  Each statement generates a new schema based on the schema to its left (or the base, for the first command).  These transformations overwrite previous annotations in their path, i.e. if your dataset contains `va.a: Int` and you insert `va.a.b = 10`, `va.a: Int` is overwritten.  These overwrites happen left to right: 

```
annotatevariants -c 'va.a = 1, va.a = 2, va.a = 3'
```

The above command will produce one annotation, where `va.a` is `3`.  You should not do this because it is silly.