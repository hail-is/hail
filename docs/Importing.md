# Importing data into Hail

Hail does not operate directly on VCF files.  Hail uses a fast and storage-efficient internal representation called a VDS (variant dataset).  In order to use Hail for data analysis, data must first be imported to the VDS format.  This is done with the `import` command.  Hail is designed to be maximally compatible with files in the [VCF v4.2 spec](https://samtools.github.io/hts-specs/VCFv4.2.pdf).

Command line arguments: 
 - `-i` -- path of input file (required)
 - `-d` -- do not compress VDS, not recommended (optional)
 - `-n` -- number of partitions, advanced user option (optional)
 - `-f` -- force load `.gz` file, not recommended (see below) (optional)

## Importing VCF files with the import command

 - Ensure that the VCF file is correctly prepared for import:
   - VCFs should be either uncompressed (".vcf") or block-compressed (".vcf.bgz").  If you have a large compressed VCF that ends in ".vcf.gz", it is likely that the file is actually block-compressed, and you should rename the file to ".vcf.bgz" accordingly.  If you actually have a standard gzipped file, it is possible to import it to hail using the `-f` option.  However, this is not recommended -- all parsing will have to take place on one node, because gzip unzipping is not parallelizable.  In this case, import will take more than 2 orders of magnitude longer.
   - VCFs should be mounted to the hadoop file system
 - Run a hail command with `import`.  The below command will read a vcf.bgz file and write to a vds file (hail's preferred format).  It is possible to import and operate directly on a vcf file without first doing an import/write step, but this will greatly increase compute time if multiple commands are run (it is about 100 times faster to read a vds than import a vcf).
``` 
$ hail import -i /path/to/file.vcf.bgz write -o /path/to/output.vds
```
 - Hail makes certain assumptions about the genotype fields, see [Representation](https://github.com/broadinstitute/hail/blob/master/docs/Representation.md).  On import, Hail filters (sets to no-call) any genotype that violates these assumptions.  Hail interpets the format fields: GT, AD, OD, DP, GQ, PL; all others are silently dropped.
