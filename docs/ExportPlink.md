# Exporting to Plink files in Hail

Hail contains an `exportplink` module which will write out the internal VDS to a {.bed, .bim, .fam} fileset in the [Plink2 spec](https://www.cog-genomics.org/plink2/formats).

Command line arguments:
 - `-o | -output <file base>` -- path of output base, will write to `<file base>.bed`, `<file base>.bim`, `<file base>.fam`
 - `-t | --tmpdir <directory>` -- path of temporary directory (see below)
 - `-c | --cutoff <minGQ>` -- set to missing all genotypes below the specified GQ

Example `exportplink` command:
```
hail read -i /path/to/file.vds exportplink -o /path/to/plinkfiles -t /tmp/ --cutoff 20
```

Hail's output is designed to mirror Plink's own VCF conversion using the following command:
```
plink --vcf ~/hail/src/test/resources/sample.vcf --make-bed --out sample --const-fid --keep-allele-order
```
All differences between Hail's output and Plink's are enumerated below.
 - **.bed file**: equivalent within sample ordering
 - **.fam file:**: agrees when Plink is run with `--const-fid` argument (FID is set to "0")
 - **.bim file:**: ID field will be different.  The above Plink command will copy the "RSID" field of the VCF into the .bim ID column, leaving huge numbers of IDs as ".".  Instead, Hail will encode each variant with an ID in the format "CHR:POS:REF:ALT".