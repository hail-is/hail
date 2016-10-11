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
   
#### Designating output with .columns files

Hail supports reading in a file ending with ".columns" to assign column names and expressions.  This file should contain one line per desired column.  Each line should contain two fields, separated by a tab: the header name in the first, the expression in the second.  Below is an example of a .columns file:

```
$ cat exportSamples.columns
SAMPLE	s
CALL_RATE	sa.qc.callRate
NHET	sa.qc.nHet
```
 
The corresponding command line using the .columns file from above is as follows:

```
exportsamples -c 'exportSamples.columns' -o file.tsv
```

#### Designating output with an expression

Much like [filtering](reference.html#Filtering) modules, exporting allows flexible expressions to be written on the command line.  While the filtering modules expect an expression that evaluates to true or false, export modules expect a comma-separated list of fields to print.  These fields should take the form `IDENTIFIER = <expression>`.  Below is an example of `exportsamples` with an export expression:

```
exportsamples -c 'SAMPLE = s, CALL_RATE = sa.qc.callRate, NHET = sa.qc.nHet' -o file.tsv
```

Note that the above expression will result in identical output to the example .columns file above.

It is also possible to export without identifiers, which will result in a file with no header.  In this case, the expressions should look like the examples below:
```
exportsamples -c 's.id, sa.qc.rTiTv' -o file.tsv
```

**Note:** if some fields have identifiers and some do not, Hail will throw an exception.  Either each field must be identified, or each field should include only an expression.

</div>