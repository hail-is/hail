## <a class="jumptarget" name="filtering"></a> Filtering Data

Hail provides a number of commands to filter data. See the [Hail documentation on Filtering](reference.html#Filtering) for an overview of available commands.

#### How do I create a sites-only VCF (all genotype data removed)?

Use the [`filtersamples all`](commands.html#filtersamples_all) command to remove all genotype and sample meta-information and write the output as a VCF with the [`exportvcf`](commands.html#exportvcf) command. **Make sure to add the ".vcf.bgz" extension to your filename to ensure the VCF file is block compressed!**

```
filtersamples all 
exportvcf -o /path/to/sitesonly.vcf.bgz
```


#### How do I filter out samples listed in a text file?

Use the [`filtersamples list`](commands.html#filtersamples_list) command, which takes as input a text file with one sample ID per line. Equivalent to `--remove` in PLINK.

```
filtersamples list --remove -i /path/to/badSamples.sample_list
```


#### How do I only keep the samples listed in a text file?

Use the [`filtersamples list`](commands.html#filtersamples_list) command, which takes as input a text file with one sample ID per line. Equivalent to `--keep` in PLINK.

```
filtersamples list --keep -i /path/to/goodSamples.sample_list
```


#### How do I subset the dataset to one chromosome?

Use the [`filtervariants expr`](commands.html#filtervariants_expr) command, which takes as input a boolean expression. 
To access the chromosome name, use the `v.contig` method, which results in a `String` type. 
Double quotes around the chromosome name are necessary to ensure Hail interprets the chromosome as a string. 
Equivalent to `--chr 1` in PLINK.

```
filtervariants expr --keep -c 'v.contig == "1"'
```


#### How do I filter the VDS to a subset of chromosomes?

To subset on multiple chromosomes, use the `let` function in the [Hail Expression Language](reference.html#HailExpressionLanguage) to create a set containing the desired chromosome values and then check whether `v.contig` is in the set. 
Equivalent to `--chr 1, 2, 3` in PLINK

```
filtervariants expr --keep -c 'let chrom=set("1","2","3") in 
    chrom.contains(v.contig)'
```


#### How do I filter the VDS to not include sex chromosomes (autosomes only)?

Equivalent to `--not-chr X, Y, XY` or `--autosome` in PLINK. 

Chromosome names have the type `String` in Hail.

If the data was imported from a VCF, the chromosome names are the same as the first column of the original VCF file.

If the data was imported from a PLINK, BGEN, or GEN file, then chromosome codes have been converted as follows:

 - 23 -> "X"
 - 24 -> "Y"
 - 25 -> "X"

```
filtervariants expr --keep -c 'v.contig != "X" && v.contig != "Y"'
```


#### How do I subset the dataset to only include non-pseudoautosomal variants on the X chromosome?

Equivalent to `--chr 23` in PLINK. 

When using Hail's `v.inParX` method, Hail assumes the genome build is HG19 and the base pair boundaries of the pseudoautosomal region is 60001-2699520 and 154931044 - 155260560.

References for pseudoautosomal regions:

 - [Wikipedia article on PAR](https://en.wikipedia.org/wiki/Pseudoautosomal_region)
 - [Boundaries for build GRCh37 from NCBI](http://www.ncbi.nlm.nih.gov/projects/genome/assembly/grc/human/)

```
filtervariants expr --keep -c 'v.contig == "X" && !v.inParX'
```


#### How do I remove samples from my dataset based on summary statistics calculated from sample QC?

Example: Remove samples with a call rate less than 95% 

First use the [`sampleqc`](commands.html#sampleqc) command to calculate the call rate per sample.
The output of the [`sampleqc`](commands.html#sampleqc) command will be stored as a sample annotation that can be accessed with `sa.qc.callRate`.
Lastly, use the [`filtersamples expr`](commands.html#filtersamples_expr) command with the `--keep` flag to only keep samples where `sa.qc.callRate` is greater than a desired cutoff.
 
```
sampleqc 
filtersamples expr --keep -c 'sa.qc.callRate >= 0.95'
```

See the [documentation for `sampleqc`](commands.html#sampleqc) for additional sample QC summary statistics that can be used for filtering.


#### How do I remove variants from my dataset based on summary statistics calculated from variant QC?

Example: Remove variants with a call rate less than 95% 

First use the [`variantqc`](commands.html#variantqc) command to calculate the call rate per variant.
The output of the [`variantqc`](commands.html#variantqc) command will be stored as a variant annotation that can be accessed with `va.qc.callRate`.
Lastly, use the [`filtervariants expr`](commands.html#filtervariants_expr) command with the `--keep` flag to only keep variants where `va.qc.callRate` is greater than a desired cutoff.
 
```
variantqc 
filtervariants expr --keep -c 'va.qc.callRate >= 0.95 && 
    va.qc.dpMean >= 20 && va.qc.pHWE > 1e-6 && va.pass'
```

See the [documentation for `variantqc`](commands.html#variantqc) for additional variant QC summary statistics that can be used for filtering.


#### How do I filter genotypes using genotype meta-information such as PL or DP?

Use the [`filtergenotypes`](commands.html#filtergenotypes) command which takes a boolean expression as input and either a `--keep` or `--remove` flag. 
The PL and DP fields can be accessed with `g.pl` and `g.dp` respectively. 
Other available fields can be found in the [Documentation for the Genotype object](reference.html#genotype).
See the documentation of the [Hail Expression Language](reference.html#HailExpressionLanguage) for additional information on creating expressions.

``` 
filtergenotypes --remove -c 'g.dp < 10 || 
   (g.ad[0] + g.ad[1]) / g.dp < 0.9 || 
   (g.isHomRef && (g.ad[0] / g.dp < 0.9 || g.gq < 25)) ||
   (g.isHet && (g.ad[1] / g.dp < 0.25 || g.pl[0] < 25)) ||
   (g.isHomVar && (g.ad[1] / g.dp < 0.9 || g.pl[0] < 25))'
```

