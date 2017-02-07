<div class="cmdhead"></div>

<div class="description"></div>

<div class="synopsis"></div>

<div class="options"></div>

<div class="cmdsubsection">

### Notes:

 - Hail supports importing data in the BGEN file format. For more information on the BGEN file format, see [here](http://www.well.ox.ac.uk/~gav/bgen_format/bgen_format_v1.1.html). **Only v1.1 BGEN files are supported at this time**.

 - Ensure that the BGEN file(s) and Sample File are correctly prepared for import:
    - Files should reside in the hadoop file system
    - The sample file should have the same number of samples as the BGEN file
    - No duplicate sample IDs are allowed
    
 - The sample id `s.id` used is the first column in the .sample file
  
 - Chromosome codes 23, 24, and 25 are mapped to "X", "Y", and "X", respectively.
  
#### Dosage representation:
 - Hail automatically filters out any genotypes where the absolute value of the sum of the dosages is greater than a certain tolerance (specified by `-t` or `--tolerance`) from 1.0. The default value is 0.02.
 - Hail normalizes all dosages to sum to 1.0. Therefore, an input dosage of (0.98, 0.0, 0.0) will be stored as (1.0, 0.0, 0.0) in Hail.
 - Hail will give slightly different results than the original data (maximum difference observed is 3E-4). 

</div>
 
<div class="cmdsubsection">

### Annotations:

Name | Type | Description
--- | :-: | ---
`va.varid` |   `String` | if a chromosome field is present, the 2nd column of the .gen file (otherwise, the 1st column of the .gen file)
`va.rsid`  |        `String` | if a chromosome field is present, the 3rd column of the .gen file (otherwise, the 2nd column of the .gen file)

</div>

<div class="cmdsubsection">

### Examples:

<h4 class="example">Import data from a single BGEN and sample file</h4>

To import data, first run [`indexbgen`](#indexbgen) and then `importbgen`.  The below command will first create an index for the .bgen file, then read the .bgen and a .sample files, and lastly write to a .vds file (Hail's preferred format).
``` 
$ hail indexbgen /path/to/file.bgen 
    importbgen -s /path/to/file.sample /path/to/file.bgen write -o /path/to/output.vds
```
 
<h4 class="example">Import data from multiple BGEN files and a sample file</h4>

To import data, first run [`indexbgen`](#indexbgen) and then `importbgen`.  The below command will first create an index for the .bgen file, then read the .bgen and a .sample files, and lastly write to a .vds file (Hail's preferred format).

To load multiple files at the same time, use [Hadoop glob patterns](reference.html#hadoopglob):
``` 
$ hail indexbgen /path/to/file.chr*.bgen
    importbgen -s /path/to/file.sample /path/to/file.chr*.bgen write -o /path/to/output.vds
```

</div>