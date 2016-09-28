<div class="cmdhead"></div>

<div class="description"></div>

<div class="synopsis"></div>

<div class="options"></div>

<div class="cmdsubsection">

### Notes:

This command allows you to compute statistics for a set of genomic intervals.

Intervals are **left inclusive, right exclusive**.  This means that \[chr1:1, chr1:3\) contains chr1:1 and chr1:2.

#### Designating output with an expression:
An export expression designates a list of computations to perform, and what these columns are named.  These expressions should take the form `COL_NAME_1 = <expression>, COL_NAME_2 = <expression>, ...`.  See the examples for ideas.

**Namespace in the `aggregateintervals` condition:**

Identifier | Description
:-: | ---
`interval` | genomic interval, see the [representation docs](reference.html#Representation) for details
`global` | global annotation
`variants` | Variant [aggregable](reference.html#aggregables).  Aggregator namespace below.


**Namespace within `variants` aggregator:**

Identifier | Description
:-: | ---
`v` | Variant
`va` | Variant annotations
`global` | Global annotations

</div>

<div class="cmdsubsection">

<h4 class="example">Count the total number of SNPs, indels, and total variants in an inteval</h4>

In this example, we have an interval file as below:

```
$ cat capture_intervals.txt
4:1500-123123
5:1-1000000
16:29500000-30200000
```

Perhaps we want to calculate the total number of SNPs, indels, and total variants in these intervals.  We can do this with the following invocation:

```
$ hail 
    read dataset.vds
    aggregateintervals
        -i capture_intervals.txt
        -o out.txt
        -c 'n_SNP = variants.filter(v => v.altAllele.isSNP).count(), n_indel = variants.filter(v => v.altAllele.isIndel).count(), n_total = variants.count()'
```

This will write out the following file:

```
Contig    Start       End         n_SNP   n_indel     n_total
4         1500        123123      502     51          553
5         1           1000000     25024   4500        29524
16        29500000    30200000    17222   2021        19043
```

The `-c` argument both defines the names of the column headers (n_SNP, n_indel, n_total) as well as the calculations for each interval.

<h4 class="example">Count the number of LOF, missense, and synonymous non-reference calls per interval</h4>


```
$ cat intervals.txt
4:1500-123123
5:1-1000000
16:29500000-30200000
```

```
$ hail 
    read dataset.vds
    annotatevariants expr -c 'va.n_calls = gs.filter(g.isCalledNonRef).count()'
    aggregateintervals
        -i intervals.txt
        -o out.txt
        -c 'LOF_CALLS = variants.filter(v => va.consequence == "LOF").map(v => va.n_calls).sum(),
            MISSENSE_CALLS = variants.filter(v => va.consequence == "missense").map(v => va.n_calls).sum(),
            SYN_CALLS = variants.filter(v => va.consequence == "synonymous").map(v => va.n_calls).sum()'
```

We will get something that looks like the following output:

```
Contig    Start       End         LOF_CALLS   MISSENSE_CALLS   SYN_CALLS
4         1500        123123      42          122              553
5         1           1000000     3           12               66
16        29500000    30200000    17          22               202
```

</div>