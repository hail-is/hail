<div class="cmdhead"></div>

<div class="description"></div>

<div class="synopsis"></div>

<div class="options"></div>

<div class="cmdsubsection">
### Notes:
 
This module takes a condition argument (`-c`) similar to [filtering](reference.html#Filtering) expressions, with a similar namespace as well.  However, the expression is not parsed as a boolean, but rather a comma-delimited list of fields or expressions to print.  These fields will be printed in the order they appear in the expression in the header and on each line.

One line per sample in the VDS will be printed.  The accessible namespace includes:

   - `s` (sample)
   - `sa` (sample annotations)
   - `global` (global annotations)
   - `gs` (genotype column [aggregable](reference.html#aggregables))

#### Designating output with an expression

Much like [filtering](#Filtering) modules, exporting allows flexible expressions to be written on the command line.  While the filtering modules expect an expression that evaluates to true or false, export modules expect a comma-separated list of fields to print.  These fields should take the form `IDENTIFIER = <expression>`.  Below is an example of `exportsamples` with an export expression:

```
exportsamples -c 'SAMPLE = s, CALL_RATE = sa.qc.callRate, NHET = sa.qc.nHet' -o file.tsv
```

It is also possible to export without identifiers, which will result in a file with no header.  In this case, the expressions should look like the examples below:
```
exportsamples -c 's.id, sa.qc.rTiTv' -o file.tsv
```

**Note:** Either all fields must be named, or no field must be named.

Much like [filtering](reference.html#Filtering) modules, exporting allows flexible expressions to be written on the command line.  While the filtering modules expect an expression that evaluates to true or false, export modules expect a comma-separated list of fields to print.  These fields should take the form `IDENTIFIER = <expression>`.  Below is an example of `exportsamples` with an export expression:

```
exportsamples -c 'sample = s.id, phenotype = sa.fam.phenotype, PC1 = sa.pca.PC1, PC2 = sa.pca.PC2' -o file
```

In the common case that a group of annotations needs to be exported (for example, the annotations produced by `sampleqc`), one can use the `struct.*` syntax.  This syntax produces one column per field in the struct, and names them according to the struct field name.  

For example, the following invocation:

```
exportsamples -c 'sample = s.id, sa.qc.*' -o file.tsv
```

...will produce the following set of columns:

```
sample  callRate  nCalled  nNotCalled  nHomRef  ...
```

Note that using the `.*` syntax always results in named arguments, so it is not possible to export header-less files in this manner.  However, naming the "splatted" struct will apply the name in front of each column like so:

```
exportsamples -c 'sample = s.id, QC = sa.qc.*' -o file.tsv
```

```
sample  QC.callRate  QC.nCalled  QC.nNotCalled  QC.nHomRef  ...
```

</div>