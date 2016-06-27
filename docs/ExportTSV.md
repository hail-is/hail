# Exporting to TSV

Hail has three export modules which write to TSVs:
 - `exportsamples`
 - `exportvariants`
 - `exportgenotypes`
 
These three export modules take a condition argument (`-c`) similar to [filtering](Filtering.md) expressions, with a similar namespace as well.  However, the expression is not parsed as a boolean, but rather a comma-delimited list of fields or expressions to print.  These fields will be printed in the order they appear in the expression in the header and on each line.

Command line arguments: 
 - `-c <cond>, --condition <cond>` -- export expression (see below) or .columns file
 - `-o <file>, --output <file>` -- file path to which output should be written

## Export modules

1. `exportsamples` will print one line per sample in the VDS.  The accessible namespace includes:
   - `s` (sample)
   - `sa` (sample annotations)
   - `global` (global annotations)
   - `gs` (genotype column [aggregable](HailExpressionLanguage.md#aggregables))
2. `exportvariants` will print one line per variant in the VDS.  The accessible namespace includes:
   - `v` (variant)
   - `va` (variant annotations)
   - `global` (global annotations)
   - `gs` (genotype row [aggregable](HailExpressionLanguage.md#aggregables))
3. `exportgenotypes` will print one line per cell (genotype) in the VDS<sup>*</sup>.  The accessible namespace includes:
   - `g` (genotype)
   - `s` (sample)
   - `sa` (sample annotations)
   - `v` (variant)
   - `va` (variant annotations).
   - `global` (global annotations)
   
<sup>*</sup>The `exportgenotypes` module does not print hom-ref or missing genotypes by default, in order to restrict the size of the file produced.  There are command line arguments to turn on these print modes:
 - `--print-ref`
 - `--print-missing`
   
## Designating output with .columns files

Hail supports reading in a file ending with ".columns" to assign column names and expressions.  This file should contain one line per desired column.  Each line should contain two fields, separated by a tab: the header name in the first, the expression in the second.  Below are two examples of acceptable columns files:

```
$ cat exportVariants.columns
VARIANT	v
PASS	va.pass
FILTERS	va.filters
MISSINGNESS	1 - va.qc.callRate
```

```
$ cat exportGenotypes.columns
SAMPLE	s
VARIANT	v
GQ	g.gq
DP	g.dp
ANNO1	va.MyAnnotations.anno1
ANNO2	va.MyAnnotations.anno2
```
 
Appropriate command line usages using these .columns files are as follows:

```
exportvariants -c 'exportVariants.columns' -o file.tsv
```

```
exportgenotypes -c 'exportGenotypes.columns' -o file2.tsv
```

## Designating output with an expression

Much like [filtering](Filtering.md) modules, exporting allows flexible expressions to be written on the command line.  While the filtering modules expect an expression that evaluates to true or false, export modules expect a comma-separated list of fields to print.  These fields should take the form `IDENTIFIER = <expression>`.  Below are examples of acceptable export expressions:

```
exportvariants -c 'VARIANT = v, PASS = va.pass, FILTERS = va.filters, MISSINGNESS = 1 - va.qc.callRate' -o file.tsv
```

```
exportgenotypes -c 'SAMPLE=s,VARIANT=v,GQ=g.gq,DP=g.dp,ANNO1=va.MyAnnotations.anno1,ANNO2=va.MyAnnotations.anno2' -o file.tsv
```

Note that the above two expressions will result in identical output to the example .columns files above.

It is also possible to export without identifiers, which will result in a file with no header.  In this case, the expressions should look like the examples below:
```
exportsamples -c 's.id, sa.qc.rTiTv' -o file.tsv
```
```
exportvariants -c 'v,va.pass,va.qc.AF' -o file.tsv
```
```
exportgenotypes -c 'v,s.id,g.gq' -o file.tsv
```

**Note:** if some fields have identifiers and some do not, Hail will throw an exception.  Either each field must be identified, or each field should include only an expression.
