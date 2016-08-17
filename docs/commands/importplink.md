<div class="cmdhead"></div>

<div class="description"></div>

<div class="synopsis"></div>

<div class="options"></div>

<div class="cmdsubsection">
### Notes:

 - Hail supports importing data from [PLINK binary files](http://pngu.mgh.harvard.edu/~purcell/plink/data.shtml#bed). Only binary SNP-major mode files can be read into Hail. To convert your file from Individual-major mode to SNP-major mode, use PLINK to read in your fileset and use the `--make-bed` option.

 - You must use `--bfile` only, or all three of `--bed`, `--bim`, and `--fam`.

 - The centiMorgan position is not currently used in Hail (Column 3 in .bim file).
 
 - The ID (`s.id`) used by Hail is the individual ID (column 2 in .fam file).
 
 - No duplicate individual IDs are allowed.
 
 - Chromosome names (Column 1) are automatically converted in the following cases:
    - 23 => "X"
    - 24 => "Y"
    - 25 => "X"
    - 26 => "MT"
</div>

<div class="cmdsubsection">
### Examples:

<h4 class="example">Importing a Binary PLINK file with the --bfile flag</h4>
```
hail importplink --bfile /path/to/myfileroot write -o /path/to/myfile.vds
```

<h4 class="example">Importing a Binary PLINK file with the --bed, --bim, and --fam flags</h4>
```
hail importplink --bed /path/to/myfileroot.bed --bim /path/to/myfileroot.bim --fam /path/to/myfileroot.fam write -o /path/to/myfile.vds
```
</div>

