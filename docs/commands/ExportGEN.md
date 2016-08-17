<div class="cmdhead"></div>

<div class="description"></div>

<div class="synopsis"></div>

<div class="options"></div>

<div class="cmdsubsection">
### Notes:

 - Writes out the internal VDS to a {.gen, .sample} fileset in the [Oxford spec](http://www.stats.ox.ac.uk/%7Emarchini/software/gwas/file_format.html).

 - The first 6 columns of the resulting .gen file are the following:
    - Chromosome (`v.contig`)
    - Variant ID (`va.varid` if exists, else Chromosome:Position:Ref:Alt)
    - rsID (`va.rsid` if exists, else ".")
    - position (`v.start`)
    - reference allele (`v.ref`)
    - alternate allele (`v.alt`)
    
 - Probability Dosages:
    - 3 probabilities per sample (pHomRef, pHet, pHomVar)
    - Any filtered genotypes will be output as (0.0, 0.0, 0.0)
 
 - The sample file has 3 columns:
    - ID_1 and ID_2 are identical and set to the sample ID (`s.id`)
    - The third column ("missing") is set to 0 for all samples 
    
</div>

<div class="cmdsubsection">

<h4 class="example"> Export the current VDS dataset to a GEN and sample file</h4>
```
hail read -i /path/to/file.vds exportgen -o /path/to/mygenfile
```

</div>