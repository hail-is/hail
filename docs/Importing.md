# Importing Genotype Data

Hail does not operate directly on input files.  Hail uses a fast and storage-efficient internal representation called a VDS (variant dataset). In order to use Hail for data analysis, data must first be imported to the VDS format using one of the commands below. It is possible to import and operate directly on an input file without first doing an import/write step, but this will greatly increase compute time if multiple commands are run (it is significantly faster to read a vds than import a file).

### Supported Data Types:

Hail Command | Extensions | File Spec
--- | :-: | ---
[`importvcf`](#importvcf) | .vcf .vcf.bgz .vcf.gz     | [VCF file](https://samtools.github.io/hts-specs/VCFv4.2.pdf)
[`importplink`](#importplink) | .bed .bim .fam | [PLINK binary dataset](http://pngu.mgh.harvard.edu/~purcell/plink/data.shtml#bed)
[`importgen`](#importgen) | .gen .sample     | [GEN file](http://www.stats.ox.ac.uk/%7Emarchini/software/gwas/file_format.html#mozTocId40300)
[`importbgen`](#importbgen) | .bgen .sample     | [BGEN file](http://www.well.ox.ac.uk/~gav/bgen_format/bgen_format_v1.1.html)


### <a name="hadoopglob"></a> Hadoop Glob Patterns:
All of these commands take a list of files to load. Files can be specified as Hadoop glob patterns:

Character | Description
--- | :-: | ---
`?` | Matches any single character.
`*` | Matches zero or more characters.
`[abc]` | Matches a single character from character set {a,b,c}.
`[a-b]` | Matches a single character from the character range {a...b}. Note that character a must be lexicographically less than or equal to character b.
`[^a]`  | Matches a single character that is not from character set or range {a}. Note that the ^ character must occur immediately to the right of the opening bracket.
`\c`  | Removes (escapes) any special meaning of character c.
`{ab,cd}` | Matches a string from the string set {ab, cd}.
`{ab,c{de,fh}}` | Matches a string from the string set {ab, cde, cfh}.