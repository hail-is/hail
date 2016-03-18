# Splitting Multiallelic Variants

Hail has infrastructure for representing multiallelic variants, but
most analytic methods only support analyzing data represented as
biallelics.  Therefore, the current recommendation is to split
multiallelics using the command `splitmulti` when import a VCF.

Command line options:
 - `--propagate-gq` -- Propagate GQ instead of computing from PL.  Intended for use with the Michigan GotCloud calling pipeline which stores PLs but sets the GQ to the quality of the posterior probabilities.  This option is experimental and will be removed when Hail supports posterior probabilities (PP).

Example `splitmulti` command:
```
$ hail importvcf /path/to/file.vcf splitmulti write -o /path/to/file.vds
```

Methods that don't support multiallelics will generate a error message
if applied to a dataset that has not been split.  The list of commands
that support multiallelics are:
 - `count`
 - `exportvariants`
 - `exportvcf`
 - `filtersamples'
 - `filtervariants'
 - `grep`
 - `read`
 - `renamesamples`
 - `repartition`
 - `showannotations`
 - `splitmulti`
 - `vep`, and
 - `write`.

## How Hail Splits Multallelics

We will explain by example.  Consider a hypothetical 3-allelic variant
```
A	C,T	0/2:7,2,6:15:45:99,50,99,0,45,99
```

`splitmulti` will create two biallelic variants (one for each
alternate allele) at the same position:
```
A	C	0/0:13,2:15:45:0,45,99
A	T	0/1:9,6:15:50:50,0,99
```

Each multiallelic GT field is downcoded once for each alternate
allele.  A call for an alternate allele maps to 1 in the biallelic
variant corresponding to itself and 0 otherwise.  For example, in the
example above, 0/2 maps to 0/0 and 0/1.  The genotype 1/2 maps to 0/1
and 0/1.

The biallelic alt AD entry is just the multiallelic AD entry
corresponding to the alternate allele.  The ref AD entry is the sum of
the other multiallelic entries.

The biallelic DP is the same as the multiallelic DP.

The biallelic PL entry for for a genotype `g` is the minimum over PL
entries for multiallelic genotypes that downcode to `g`.  For example,
the PL for (A, T) at 0/1 is the minimum of the PLs for 0/1 (50) and
1/2 (45), and thus 45.

Fixing an alternate allele and biallelic variant, downcoding gives a
map from multiallelic to biallelic alleles and genotypes.  The
biallelic AD entry for an allele is just the sum of the multiallelic
AD entries for alleles that map to that allele.  Similarly, the
biallelic PL entry for a genotype is the minimum over multiallelic PL
entries for genotypes that map to that genotype.

By default, GQ is recomputed from PL.  If `--propagate-gq` is used,
the biallelic GQ field is simply the multiallelic GQ field, that is,
genotype qualities are unchanged.

Here is a second example for a het non-ref:
```
A	C,T	1/2:2,8,6:16:45:99,50,99,45,0,99
```
splits as
```
A	C	0/1:8,8:16:45:45,0,99
A	T	0/1:10,6:16:50:50,0,99
```
