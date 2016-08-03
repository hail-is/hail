# Importing Data from PLINK Binary Files

Hail supports importing data from [PLINK binary files](http://pngu.mgh.harvard.edu/~purcell/plink/data.shtml#bed). Only binary SNP-major mode files can be read into Hail. To convert your file from Individual-major mode to SNP-major mode, use PLINK to read in your fileset and use the `--make-bed` option.

## Command line options:
Flag | Description
--- | :-: | ---
`--bfile <file base>` | Path of input file base. Will expect `<file base>.bed`, `<file base>.bim`, `<file base>.fam` all exist
`--bed <bed file>` | Path of input .bed file 
`--bim <bim file>` | Path of input .bim file 
`--fam <fam file>` | Path of input .fam file 
`-d | --no-compress` | Do not compress VDS. Not recommended.
`-n <N> | --npartitions <N>` | Number of partitions. Advanced user option.

 You must use `--bfile` only, or all three of `--bed`, `--bim`, and `--fam`.

## Importing PLINK files with the importplink command

Example `importplink` command and writing to a VDS file:
```
hail importplink --bfile /path/to/myfileroot write -o /path/to/myfile.vds
```
or
```
hail importplink --bed /path/to/myfileroot.bed --bim /path/to/myfileroot.bim --fam /path/to/myfileroot.fam write -o /path/to/myfile.vds
```

## Assumptions:
 - The centiMorgan position is not currently used in Hail (Column 3 in .bim file).
 - The ID (`s.id`) used by Hail is the individual ID (column 2 in .fam file).
 - No duplicate individual IDs are allowed.
 - Chromosome names (Column 1) are automatically converted in the following cases:
    - 23 => "X"
    - 24 => "Y"
    - 25 => "X"
    - 26 => "MT"
