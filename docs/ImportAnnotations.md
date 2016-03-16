# Importing annotations into Hail

Hail includes modules for importing annotations from external files, for use in downstream filtering and analysis.  There are two such modules, one for annotating samples and one for annotating variants:
 - `annotatesamples`
 - `annotatevariants`
 
Most command line arguments differ between the two modules and individual file formats.  The following are universal:
 - `-c <condition or file>, --condition <condition or file>` -- path of file to read
 - `-r | --root <path>` -- root annotations in `path`.  If no root is provided, all annotations will be placed directly under `va` or `sa`.  For arbitrarily deep placement, you can specify a path like `--root va.this.is.a.big.tree` 

## Annotating Samples

There is currently one file type supported for annotating samples: 

1. **Tab separated values (tsv, tsv.gz).**  This file requires one column containing sample IDs as present in the imported vcf, and every other column in the file will be written to sample annotations.  The command line arguments for this format are as follows:
 - `-c <path-to-tsv>, --condition <path-to-tsv>`: **(Required)** specify the file path
 - `-s <id>, --sampleheader <id>` **(Optional with default)** specify the name of the column containing sample IDs.  (Default: `Sample`)
 - `-t <typestring>, --types <typestring>` **(Optional)** specify data types of fields, in a comma-delimited string of `name: Type` elements.  If a field is not found in this type map, it will be read and stored as a string. (optional)
 - `-m <missingstring>, --missing <missingstring>` **(Optional with default)** specify identifiers to be treated as missing, in a comma-separated list.  (Default: `NA`)
 
## Annotating Variants

There are currently five file types supported for annotating variants:

1. **Tab separated values (tsv, tsv.gz).**  This file format **requires** 4 columns for contig, position, ref, and alt.  Each other column in the file will be written to variant annotations.  The following command line arguments exist for .tsv files:
 - `-v | --vcolumns <columns>` -- specify the column headers for the contig, position, ref, and alt fields.  (Default: `Chromosome,Position,Ref,Alt`)
 - `-t | --types <type string>` -- specify data types of fields, in a comma-delimited string of `name:Type` elements.  If a field is not found in this type map, it will be interpreted as a string. (optional)
 - `-m | --missing <missing values>` -- specify identifiers to be treated as missing, in a comma-separated list.  (Default: `NA`)
2. **VCF (vcf, vcf.gz, vcf.bgz).**  This file format **requires** the `--root` command line option so that all info field annotations in the variant dataset are not overwritten.
4. **Interval list (.interval_list, .interval_list.gz).**  This file extension encompasses two file formats, `chr:start-end` and `chr start end strand target` (tsv).  The former will produce a boolean annotation, while the latter will store the `target` as a string.  The following argument is **required** for interval_list files:
 - `-i | --identifier <name>` -- Choose the name of the annotation in the vds.  If a `root` is specified, it can be found in `va.root.identifier`, otherwise `va.identifier`.
5. **UCSC bed (.bed, .bed.gz).**  This format is similar to the interval_list format.  The annotation name is designated in the track header of the bed file (`name="identifier"`).  If the body of the file contains the fourth (name) column, the annotation will be stored as a string with that field, otherwise boolean.  The spec for UCSC BED files is defined  [here.](https://genome.ucsc.edu/FAQ/FAQformat.html#format1)
6. **Hail-processed RDDs (.faf).**  Large TSV and VCF files can be very slow to parse and load into memory.  Since we still want to load these files, Hail supports reading pre-parsed and serialized files generated with the `convertannotations` module. [See conversion documentation here.](ConvertAnnotations.md)