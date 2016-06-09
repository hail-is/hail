# Imputing Sex in Hail

Hail contains an `imputesex` module which will calculate the inbreeding coefficient on the X chromosome as done in [PLINK v1.7](http://pngu.mgh.harvard.edu/~purcell/plink/summary.shtml#sexcheck). All results are stored in sample annotations.

## Command line options:

Short Flag | Long Flag | Description | Default
--- | :-: | ---
`-m <Double>` | `--mafthreshold <Double>` | Minimum variant minor allele frequency | 0.0
`-e` | `--excludepar` | Exclude variants in pseudo autosomal regions (HG19) | false
`-x <Double>` | `--femalethreshold <Double>` | If the inbreeding coefficient is less than `<Double>`, then sample is called as Female | 0.2
`-y <Double>` | `--malethreshold <Double>` | If the inbreeding coefficient is greater than `<Double>`, then sample is called as Male | 0.8


## Example `imputesex` command:
```
hail read -i /path/to/file.vds imputesex -e -m 0.01 exportsamples -o /path/to/output.tsv -c "ID=s.id, F=sa.imputesex.F, ImputedSex=sa.imputesex.imputedSex"
```

To get the exact same answer as PLINK, make sure you split multi-allelic variants and use the following flags:
```
hail read -i /path/to/file.vds splitmulti imputesex
```

## Implementation details:

1. X chromosome variants are selected from the VDS: `v.contig == "X" || v.contig == "23"`
2. Variants with a minor allele frequency less than the threshold given by `--mafthreshold` are removed
3. Variants in the pseudoautosomal region `(X:60001-2699520) || (X:154931044-155260560)` are excluded if the `--excludepar` flag is set
4. The minor allele frequency (maf) per variant is calculated
5. For each variant and sample with a non-missing genotype call, `E`, the expected number of homozygotes (from population MAF), is computed as `1.0 - (2.0*maf*(1.0-maf))`
6. For each variant and sample with a non-missing genotype call, `O`, the observed number of homozygotes, is computed as `0 = heterozygote; 1 = homozygote`
7. For each variant and sample with a non-missing genotype call, `N` is incremented by 1
8. For each sample, `E`, `O`, and `N` are combined across variants
9. `F` is calculated by `(O - E) / (N - E)`
10. A sex is assigned to each sample with the following criteria: `F < 0.2 => Female (2); F > 0.8 => Male (1)`. Use `--femalethreshold` and `--malethreshold` to change this behavior.

## Available sample annotations:
The below annotations can be accessed with `sa.imputesex.<identifier>`

Identifier | Type | Description
--- | :-: | ---
`imputedSex` | `Int` | Imputed Sex: 1 = Male, 2 = Female
`F` | `Double` | Inbreeding coefficient
`T` | `Int` | Total number of variants considered
`N` | `Int` | Number of variants with a genotype call
`E` | `Double` | Expected number of homozygotes
`O` | `Double` | Observed number of homozygotes