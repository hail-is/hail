# Programmatic annotation in Hail

Hail supports programmatic annotation, which is the computation of new annotations from the data structures and functions exposed in the Hail command-line language.  This can be as simple as creating an annotation `va.missingness` = `1 - va.qc.callRate`, or it can be a complex computation involving conditionals and multiple stored annotations.

## Command syntax
 
In both `annotatesamples` and `annotatevariants`,  the syntax for creating programmatic annotations is the same.  These commands are composed of one or more annotation statements of the following structure:

```
annotatevariants -c   "va.path.to.new.annotation          =         if (va.pass) va.qc.gqMean else 0    [ , ... ]"
                                ^                         ^                     ^                           ^
                       period-delimited path         equals sign            expression               comma followed by
                     starting with 'va' or 'sa'     delimits path                                     more annotation
                                                    and expression                                       statements
```
 
In this example, `annotatevariants` will produce a new annotation stored in `va.path.to.new.annotation`, which is accessible in all future modules. 
 
## Order of operations

When you generate more than one new annotation, the operations are structured in the following order:

1.  All computations are evaluated from the current annotations.  This means that you cannot write `annotatevariants -c 'va.a = 5, va.b = va.a * 2'` -- `va.a` does not exist in the base schema, so it is not found during the computations.
2.  Each statement generates a new schema based on the schema to its left (or the base, for the first command).  These transformations existing 

For the sake of this example, we will define a dummy annotation schema:

```
    va
        a : Int
        b : Int
        c : Struct
            c1: String
            c2: String
```

In this schema, the following are accessible fields: `va.a`, `va.b`, `va.c.c1`, and `va.c.c2`.  Programmatic annotation