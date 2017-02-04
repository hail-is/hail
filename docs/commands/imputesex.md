<div class="cmdhead"></div>

<div class="description"></div>

<div class="synopsis"></div>

<div class="options"></div>

<div class="cmdsubsection">
### Notes:

We used the same implementation as [PLINK v1.7](http://pngu.mgh.harvard.edu/~purcell/plink/summary.shtml#sexcheck).

1. X chromosome variants are selected from the VDS: `v.contig == "X" || v.contig == "23"`
2. Variants with a minor allele frequency less than the threshold given by `--maf-threshold` are removed
3. Variants in the pseudoautosomal region `(X:60001-2699520) || (X:154931044-155260560)` are included if the `--include-par` flag is set
4. The minor allele frequency (maf) per variant is calculated
5. For each variant and sample with a non-missing genotype call, `E`, the expected number of homozygotes (from population MAF), is computed as `1.0 - (2.0*maf*(1.0-maf))`
6. For each variant and sample with a non-missing genotype call, `O`, the observed number of homozygotes, is computed as `0 = heterozygote; 1 = homozygote`
7. For each variant and sample with a non-missing genotype call, `N` is incremented by 1
8. For each sample, `E`, `O`, and `N` are combined across variants
9. `F` is calculated by `(O - E) / (N - E)`
10. A sex is assigned to each sample with the following criteria: `F < 0.2 => Female; F > 0.8 => Male`. Use `--female-threshold` and `--male-threshold` to change this behavior.
</div>

<div class="cmdsubsection">
### Annotations:

The below annotations can be accessed with `sa.imputesex.<identifier>`

Identifier | Type | Description
--- | :-: | ---
`isFemale` | `Boolean` | True if the imputed sex is female, false if male, missing if undetermined
`Fstat` | `Double` | Inbreeding coefficient
`nTotal` | `Long` | Total number of variants considered
`nCalled` | `Long` | Number of variants with a genotype call
`expectedHoms` | `Double` | Expected number of homozygotes
`observedHoms` | `Long` | Observed number of homozygotes

</div>

<div class="cmdsubsection">
### Examples:

<h4 class="example"> Impute the sex of samples </h4>

```
hail read /path/to/file.vds imputesex
```

<h4 class="example">Output the results to a text file</h4>

```
hail read /path/to/file.vds imputesex 
   exportsamples -o /path/to/output.tsv 
   -c "ID=s.id, Fstat=sa.imputesex.Fstat, ImputedSex=sa.imputesex.isFemale"
```

<h4 class="example">Calculate the inbreeding coefficient in only common variants (Minor Allele Frequency > 5%)</h4>

```
hail read /path/to/file.vds imputesex -m 0.05
```

<h4 class="example">Use custom thresholds when imputing sex from the inbreeding coefficient</h4>

```
hail read /path/to/file.vds imputesex 
   --male-threshold 0.5 --female-threshold 0.5
```

<h4 class="example">Include pseudoautosomal variants in the inbreeding coefficient calculation</h4>

```
hail read /path/to/file.vds imputesex --include-par
```

<h4 class="example">Use a population reference minor allele frequency when calculating the inbreeding coefficient</h4> 

```
hail read /path/to/file.vds 
   annotatevariants table -i my_exac_stats.tsv -root "va.exac"  
   imputesex --pop-freq "va.exac.maf"
```

<h4 class="example"> Obtain identical results as PLINK v1.7</h4>

```
hail read /path/to/file.vds 
   splitmulti 
   imputesex --include-par
```

<h4 class="example">Check the reported sex against the imputed sex</h4>
```
hail read /path/to/file.vds 
    annotatesamples fam -i my_cohort_phenotypes.fam -r sa.mypheno 
    imputesex 
    annotatesamples expr -c 'sa.sexcheck = sa.mypheno.isFemale == sa.imputeSex.isFemale'
```
</div>