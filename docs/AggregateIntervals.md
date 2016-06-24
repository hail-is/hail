# Using the `aggregateintervals` module

This command allows you to compute statistics for a set of genomic intervals.

**Command line arguments:**

Shortcut | Argument | Description
:-: | :-: | ---
`-i` | `--input <input file>` | Path to interval list file
`-o` | `--output <output path>` | Path to which the output file should be written
`-c` | `--condition <expression>` | Export expression (see below)

This module aggregates statistics for variants in a set of intervals.  The exposed 

**Namespace in the `aggregateintervals` module:**

Identifier | Description
:-: | ---
`interval.contig` | chromosome of interval
`interval.start` | start position of interval
`interval.end` | end position of interval
`global` | global annotation
`variants` | Variant [aggregable](HailExpressionLanguage.md#aggregables).  Aggregator namespace below.

____

**Namespace within `variants` aggregators:**

Identifier | Description
:-: | ---
`v` | Variant
`va` | Variant annotations
`global` | Global annotations

____

###Examples:

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
    read -i dataset.vds
    aggregateintervals
        -i capture_intervals.txt
        -o out.txt
        -c 'n_SNP = variants.count(v.altAllele.isSNP), n_indel = variants.count(v.altAllele.isIndel), n_total = variants.count(true)'
```

This will write out the following file:

```
Chromosome  Start       End         n_SNP   n_indel     n_total
4           1500        123123      502     51          553
5           1           1000000     25024   4500        29524
16          29500000    30200000    17222   2021        19043
```

The `-c` argument both defines the names of the column headers (n_SNP, n_indel, n_total) as well as the calculations for each interval.