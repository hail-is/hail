<div class="cmdhead"></div>

<div class="description"></div>

<div class="synopsis"></div>

<div class="options"></div>

<div class="cmdsubsection">
### Notes:

`exportplink` writes out the internal VDS to a {.bed, .bim, .fam} fileset in the [Plink2 spec](https://www.cog-genomics.org/plink2/formats).

Example `exportplink` command:
```
hail read /path/to/file.vds exportplink -o /path/to/plink
```

Hail's output is designed to mirror Plink's own VCF conversion using the following command:
```
plink --vcf /path/to/file.vcf --make-bed --out sample --const-fid --keep-allele-order
```

All differences between Hail's output and Plink's are enumerated below.
 - **.bed file**: equivalent within multiallelic split variant ordering
 - **.fam file**: agrees when Plink is run with `--const-fid` argument (FID is set to "0")
 - **.bim file**: ID field will be different.  The above Plink command will copy the "RSID" field of the VCF into the .bim ID column, leaving huge numbers of IDs as ".".  Instead, Hail will encode each variant with an ID in the format "CHR:POS:REF:ALT".

</div>