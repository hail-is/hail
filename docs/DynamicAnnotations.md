# Programmatic annotation in Hail

Hail supports programmatic annotation, which is the computation of new annotations from the data structures and functions exposed in the Hail command-line language.  This can be as simple as creating an annotation `va.missingness` = `1 - va.qc.callRate`, or it can be a complex computation involving conditionals and multiple stored annotations.

## Command syntax
 
In both `annotatesamples` and `annotatevariants`,  the syntax for creating programmatic annotations is the same.  These commands are composed of one or more annotation statements of the following structure:
```
                       va.path.to.new.annotation          =         if (va.pass) va.qc.gqMean else 0    [ , ... ]
                                ^                         ^                     ^                           ^
                       period-delimited path         equals sign            expression               comma followed by
                     starting with 'va' or 'sa'     delimits path                                     more annotation
                                                    and expression                                       statements
```
 
## Order of operations

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