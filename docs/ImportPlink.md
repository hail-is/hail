# Importing a Plink Binary File

Hail contains an `importplink` module which will read a binary PLINK fileset {.bed, .bim, .fam} in the [Plink2 spec](https://www.cog-genomics.org/plink2/formats).
Only SNP-major mode files can be read into Hail. To convert your file from Individual-major mode to SNP-major mode, use PLINK to read in your fileset and use the `--make-bed` option.

Command line options:
 - `--bfile <file base>` -- path of input file base, will expect `<file base>.bed`, `<file base>.bim`, `<file base>.fam` all exist
or
 - `--bed <bed file>` -- path of input .bed file
 - `--bim <bim file>` -- path of input .bim file
 - `--fam <fam file>` -- path of input .fam file

Example `importplink` command:
```
hail importplink --bfile /path/to/myfileroot
```
or
```
hail importplink --bed /path/to/myfileroot.bed --bim /path/to/myfileroot.bim --fam /path/to/myfileroot.fam
```

Assumptions:
 - The centiMorgan position is not currently used in Hail (Column 3 in .bim file).
 - The ID used by Hail is the individual ID (column 2 in .fam file).
 - No duplicate individual IDs are allowed.