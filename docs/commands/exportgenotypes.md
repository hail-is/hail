<div class="cmdhead"></div>

<div class="description"></div>

<div class="synopsis"></div>

<div class="options"></div>

<div class="cmdsubsection">
### Notes:
 
This module takes a condition argument (`-c`) similar to [filtering](#Filtering) expressions, with a similar namespace as well.  However, the expression is not parsed as a boolean, but rather a comma-delimited list of fields or expressions to print.  These fields will be printed in the order they appear in the expression in the header and on each line.

One line per cell (genotype) in the VDS<sup>*</sup> will be printed.  The accessible namespace includes:

   - `g` (genotype)
   - `s` (sample)
   - `sa` (sample annotations)
   - `v` (variant)
   - `va` (variant annotations)
   - `global` (global annotations)
   
<sup>*</sup>By default, hom-ref or missing genotypes are not printed in order to restrict the size of the file produced.  There are command line arguments to turn on these print modes:
 
 - `--print-ref`
 - `--print-missing`
   
#### Designating output with .columns files

Hail supports reading in a file ending with ".columns" to assign column names and expressions.  This file should contain one line per desired column.  Each line should contain two fields, separated by a tab: the header name in the first, the expression in the second.  Below is an example of an acceptable column file:

```
$ cat exportGenotypes.columns
SAMPLE	s
VARIANT	v
GQ	g.gq
DP	g.dp
ANNO1	va.MyAnnotations.anno1
ANNO2	va.MyAnnotations.anno2
```
 
The corresponding command to use this .columns file is as follows:

```
exportgenotypes -c 'exportGenotypes.columns' -o file.tsv
```

#### Designating output with an expression

Much like [filtering](#Filtering) modules, exporting allows flexible expressions to be written on the command line.  While the filtering modules expect an expression that evaluates to true or false, export modules expect a comma-separated list of fields to print.  These fields should take the form `IDENTIFIER = <expression>`.  Below is an example of using `exportgenotypes` with an export expression:

```
exportgenotypes -c 'SAMPLE=s,VARIANT=v,GQ=g.gq,DP=g.dp,ANNO1=va.MyAnnotations.anno1,ANNO2=va.MyAnnotations.anno2' -o file.tsv
```

Note that the expression above will result in identical output to the example .columns file above.

It is also possible to export without identifiers, which will result in a file with no header.  In this case, the expressions should look like the examples below:

```
exportgenotypes -c 'v,s.id,g.gq' -o file.tsv
```

**Note:** if some fields have identifiers and some do not, Hail will throw an exception.  Either each field must be identified, or each field should include only an expression.


</div>