# Exporting Genotype Data

Hail can export data from a VDS to a number of different formats listed below:

### Supported Data Types:

Hail Command | Extensions | File Spec
--- | :-: | ---
[`exportvcf`](#exportvcf) | .vcf .vcf.bgz .vcf.gz     | [VCF file](https://samtools.github.io/hts-specs/VCFv4.2.pdf)
[`exportplink`](#exportplink) | .bed .bim .fam | [PLINK binary dataset](http://pngu.mgh.harvard.edu/~purcell/plink/data.shtml#bed)
[`exportgen`](#exportgen) | .gen .sample     | [GEN file](http://www.stats.ox.ac.uk/%7Emarchini/software/gwas/file_format.html#mozTocId40300)

