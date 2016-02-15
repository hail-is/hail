# Exporting to VCF

Hail contains an `exportvcf` module which will write out the internal VDS in a vcf file to the [VCF 4.2 spec](https://samtools.github.io/hts-specs/VCFv4.2.pdf).

Command line arguments:
 - `-a <file>` -- File to append to VCF header.  Optional.
 - `-o <output file> | --output <output file>` -- Output file.  Required.

Example `exportvcf` commands:
```
hail read -i /path/to/file.vds exportvcf -o /path/to/file.vcf
```

```
hail read -i /path/to/file.vds exportvcf -o /path/to/file.vcf.bgz
```

## Information written

Importing a VCF into Hail's VDS then exporting it as a VCF will not result in exactly the same VCF.  The differences, however, are limited to the header and INFO field.  Hail's VCF header will contain:
 - FORMAT lines, with the format "GT:AD:DP:GQ:PL"
 - FILTER lines, if present in original imported VCF
 - INFO lines, if present in original imported VCF
 
Hail's VCF header will not contain:
 -  contig lines
 - lines added by external tools (bcftools, GATK, etc.)
  
Hail's INFO field will contain the same fields as the original VCF, but the ordering may be different.

Hail's VCF will not contain any added sample or variant annotations.
