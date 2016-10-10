<div class="cmdhead"></div>

<div class="description"></div>

<div class="synopsis"></div>

<div class="options"></div>

<div class="cmdsubsection">
### Notes:
 
This module takes a condition argument (`-c`) similar to [filtering](reference.html#Filtering) expressions, with a similar namespace as well.  However, the expression is not parsed as a boolean, but rather a comma-delimited list of fields or expressions to print.  These fields will be printed in the order they appear in the expression in the header and on each line.

One line per variant in the VDS will be printed.  The accessible namespace includes:

   - `v` (variant)
   - `va` (variant annotations)
   - `global` (global annotations)
   - `gs` (genotype row [aggregable](reference.html#aggregables))
   
#### Designating output with an expression

Much like [filtering](#Filtering) modules, exporting allows flexible expressions to be written on the command line.  While the filtering modules expect an expression that evaluates to true or false, export modules expect a comma-separated list of fields to print.  These fields should take the form `IDENTIFIER = <expression>`.  Below is an example of using the `exportvariants` command with an expression:

```
exportvariants -c 'VARIANT = v, PASS = va.pass, FILTERS = va.filters, MISSINGNESS = 1 - va.qc.callRate' -o file.tsv
```

It is also possible to export without identifiers, which will result in a file with no header.  In this case, the expressions should look like the examples below:
```
exportvariants -c 'v,va.pass,va.qc.AF' -o file.tsv
```

**Note:** Either all fields must be named, or no field must be named.

Much like [filtering](reference.html#Filtering) modules, exporting allows flexible expressions to be written on the command line.  While the filtering modules expect an expression that evaluates to true or false, export modules expect a comma-separated list of fields to print.  These fields should take the form `IDENTIFIER = <expression>`.  Below is an example of using the `exportvariants` command with an expression:

```
exportvariants -o file -c 'variant = v, filters = va.filters'
```

In the common case that a group of annotations needs to be exported (for example, the annotations produced by `variantqc`), one can use the `struct.*` syntax.  This syntax produces one column per field in the struct, and names them according to the struct field name.  

For example, the following invocation:

```
exportvariants -c 'variant = v, va.qc.*' -o file.tsv
```

...will produce the following set of columns:

```
variant  callRate  AC  AF  nCalled  ...
```

Note that using the `.*` syntax always results in named arguments, so it is not possible to export header-less files in this manner.  However, naming the "splatted" struct will apply the name in front of each column like so:

```
exportvariants -c 'variant = v, QC = va.qc.*' -o file.tsv
```

```
variant  QC.callRate  QC.AC  QC.AF  QC.nCalled  ...
```

</div>