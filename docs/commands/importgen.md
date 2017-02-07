<div class="cmdhead"></div>

<div class="description"></div>

<div class="synopsis"></div>

<div class="options"></div>

<div class="cmdsubsection">
### Notes:

 - Hail supports importing dosage data from a GEN file and a corresponding sample file. For more information on the GEN file format, see [here](http://www.stats.ox.ac.uk/%7Emarchini/software/gwas/file_format.html#mozTocId40300).

 - Ensure that the GEN file(s) and Sample File are correctly prepared for import:
    - Files should reside in the hadoop file system
    - If there are only 5 columns before the start of the dosage data (chromosome field is missing), you must specify the chromosome using the `-c` or `--chromosome` option
    - No duplicate sample IDs are allowed
 
 - The sample id `s.id` used is the first column in the .sample file
 
 - Chromosome codes 23, 24, and 25 are mapped to "X", "Y", and "X", respectively.
 
<a class="jumptarget" href="dosagefilters"></a> #### Dosage representation
 - Hail automatically filters out any genotypes where the absolute value of the sum of the dosages is greater than a certain tolerance (specified by `-t` or `--tolerance`) from 1.0. The default value is 0.02.
 - Hail normalizes all dosages to sum to 1.0. Therefore, an input dosage of (0.98, 0.0, 0.0) will be stored as (1.0, 0.0, 0.0) in Hail.
 - Hail will give slightly different results than the original data (maximum difference observed is 3E-4). 
</div>

<div class="cmdsubsection">
### Annotations:

Name | Type | Description
--- | :-: | ---
`va.varid` |   `String` | if a chromosome field is present, the 2nd column of the .gen file (otherwise, the 1st column of the .gen file)
`va.rsid`  |   `String` | if a chromosome field is present, the 3rd column of the .gen file (otherwise, the 2nd column of the .gen file)

</div>

<div class="cmdsubsection">
### Examples:

<h4 class="example">Read a .gen and a .sample file and write to a .vds file</h4>
``` 
$ hail importgen -s /path/to/file.sample /path/to/file.gen write -o /path/to/output.vds
```
  
<h4 class="example">Load multiple files at the same time</h4>

To specify multiple files, use [Hadoop glob patterns](reference.html#hadoopglob)
``` 
$ hail importgen -s /path/to/file.sample /path/to/file.chr*.gen write -o /path/to/output.vds
```
</div>