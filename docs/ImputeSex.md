# Imputing Sex in Hail

Hail contains an `imputesex` module which will calculate the inbreeding coefficient on the X chromosome as done in [PLINK v1.7](http://pngu.mgh.harvard.edu/~purcell/plink/summary.shtml#sexcheck). All results are stored in sample annotations.

## Command line options:

Short Flag | Long Flag | Description | Default
--- | :-: | ---
`-m <Double>` | `--maf-threshold <Double>` | Minimum variant minor allele frequency | 0.0
`-i` | `--include-par` | Include variants in pseudo autosomal regions (HG19) | false
`-x <Double>` | `--female-threshold <Double>` | If the inbreeding coefficient is less than `<Double>`, then sample is called as Female | 0.2
`-y <Double>` | `--male-threshold <Double>` | If the inbreeding coefficient is greater than `<Double>`, then sample is called as Male | 0.8
`-p <expression>` | `--pop-freq <expression>` | Population minor allele frequency (variant annotation expression, e.g. `va.exac.AF`) | *Compute from genotypes*

## Example `imputesex` command:
```
hail read -i /path/to/file.vds imputesex -m 0.01 exportsamples -o /path/to/output.tsv -c "ID=s.id, Fstat=sa.imputesex.Fstat, ImputedSex=sa.imputesex.isFemale"
```

To get the exact same answer as PLINK, make sure you split multi-allelic variants and use the flag `--include-par`:
```
hail read -i /path/to/file.vds splitmulti imputesex --include-par
```

## Implementation details:

1. X chromosome variants are selected from the VDS: `v.contig == "X" || v.contig == "23"`
2. Variants with a minor allele frequency less than the threshold given by `--mafthreshold` are removed
3. Variants in the pseudoautosomal region `(X:60001-2699520) || (X:154931044-155260560)` are included if the `--include-par` flag is set
4. The minor allele frequency (maf) per variant is calculated
5. For each variant and sample with a non-missing genotype call, `E`, the expected number of homozygotes (from population MAF), is computed as `1.0 - (2.0*maf*(1.0-maf))`
6. For each variant and sample with a non-missing genotype call, `O`, the observed number of homozygotes, is computed as `0 = heterozygote; 1 = homozygote`
7. For each variant and sample with a non-missing genotype call, `N` is incremented by 1
8. For each sample, `E`, `O`, and `N` are combined across variants
9. `F` is calculated by `(O - E) / (N - E)`
10. A sex is assigned to each sample with the following criteria: `F < 0.2 => Female; F > 0.8 => Male`. Use `--female-threshold` and `--male-threshold` to change this behavior.

## Available sample annotations:
The below annotations can be accessed with `sa.imputesex.<identifier>`

Identifier | Type | Description
--- | :-: | ---
`isFemale` | `Boolean` | True if the imputed sex is female, false if male, missing if undetermined
`Fstat` | `Double` | Inbreeding coefficient
`nVariants` | `Int` | Total number of variants considered
`nCalled` | `Int` | Number of variants with a genotype call
`expectedHoms` | `Double` | Expected number of homozygotes
`observedHoms` | `Double` | Observed number of homozygotes
