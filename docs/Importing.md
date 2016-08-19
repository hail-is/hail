# Importing Genotype Data

Hail does not operate directly on input files.  Hail uses a fast and storage-efficient internal representation called a VDS (variant dataset). In order to use Hail for data analysis, data must first be imported to the VDS format using one of the commands below. It is possible to import and operate directly on an input file without first doing an import/write step, but this will greatly increase compute time if multiple commands are run (it is significantly faster to read a vds than import a file).

### Supported Data Types:

<table>
<thead>
<tr><th>Hail Command</th><th>Extensions</th><th>File Spec</th></tr>
</thead>
<tbody>
<tr><td>[`importvcf`](#importvcf)</td><td>.vcf .vcf.bgz .vcf.gz</td><td>[VCF file](https://samtools.github.io/hts-specs/VCFv4.2.pdf)</td></tr>
<tr><td>[`importplink`](#importplink)</td><td>.bed .bim .fam</td><td>[PLINK binary dataset](http://pngu.mgh.harvard.edu/~purcell/plink/data.shtml#bed)</td></tr>
<tr><td>[`importgen`](#importgen)</td><td>.gen .sample</td><td>[GEN file](http://www.stats.ox.ac.uk/%7Emarchini/software/gwas/file_format.html#mozTocId40300)</td></tr>
<tr><td>[`importbgen`](#importbgen)</td><td>.bgen .sample</td><td>[BGEN file](http://www.well.ox.ac.uk/~gav/bgen_format/bgen_format_v1.1.html)</td></tr>
</tbody>
</table>

### <a name="hadoopglob"></a> Hadoop Glob Patterns:
All of these commands take a list of files to load. Files can be specified as Hadoop glob patterns:

<table>
<thead>
<tr><th>Character</th><th>Description</th></tr>
</thead>
<tbody>
<tr><td>`?`</td><td>Matches any single character.</td></tr>
<tr><td>`*`</td><td>Matches zero or more characters.</td></tr>
<tr><td>`[abc]`</td><td>Matches a single character from character set {a,b,c}.</td></tr>
<tr><td>`[a-b]`</td><td>Matches a single character from the character range {a...b}. Note that character a must be lexicographically less than or equal to character b.</td></tr>
<tr><td>`[^a]` </td><td>Matches a single character tat is not from character set or range {a}. Note that the ^ character must occur immediately to the right of the opening bracket.</td></tr>
<tr><td>`\c` </td><td>Removes (escapes) any special meaning of character c.</td></tr>
<tr><td>`{ab,cd}`</td><td>Matches a string from the string set {ab, cd}.</td></tr>
<tr><td>`{ab,c{de,fh}}`</td><td>Matches a string from the string set {ab, cde, cfh}.</td></tr>
</tbody>
</table>
