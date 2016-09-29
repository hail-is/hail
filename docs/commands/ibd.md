<div class="cmdhead"></div>

<div class="description"></div>

<div class="synopsis"></div>

<div class="options"></div>

<div class="subsection">

Conceptually, this command's output is a symmetric, sample-by-sample matrix. The
implementation is based on the IBD algorithm described in
[the PLINK paper](http://www.ncbi.nlm.nih.gov/pmc/articles/PMC1950838/).

This command assumes the dataset is bi-allelic. This command does not perform LD
pruning but linkage disequilibrium may negatively influence the results.

#### TSV format

The `--output` flag triggers TSV output. For example:

```
SAMPLE_ID_1	SAMPLE_ID_2	Z0	Z1	Z2	PI_HAT
sample1	sample2	1.0000	0.0000	0.0000	0.0000
sample1	sample3	1.0000	0.0000	0.0000	0.0000
sample1	sample4	0.6807	0.0000	0.3193	0.3193
sample1	sample5	0.1966	0.0000	0.8034	0.8034
    ⋮
```

#### Examples

```
... ibd --minor-allele-frequency 'va.mafs[v]' -o ibd.tsv --min 0.2 --max 0.9
```

This invocation writes only those sample pairs with `pi_hat` at or above `0.2`
and at or below `0.9` to `ibd.tsv`. It uses the provided expression to compute
the minor allele frequency.

```
... ibd -o ibd.tsv
```

This invocation writes the full IBD matrix to `ibd.tsv`. It computes the minor
allele frequency from the dataset.
