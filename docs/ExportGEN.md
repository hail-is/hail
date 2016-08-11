# Exporting a GEN file

Hail contains an `exportgen` module which will write out the internal VDS to a {.gen, .sample} fileset in the [Oxford spec](http://www.stats.ox.ac.uk/%7Emarchini/software/gwas/file_format.html).

Command line options:
 - `-o | -output <file base>` -- path of output base, will write to `<file base>.gen`, `<file base>.sample`

Example `exportgen` command:
```
hail read -i /path/to/file.vds exportgen -o /path/to/mygenfile
```

## Notes:

 - Does not support multiallelic variants. Run `splitmulti` before running `exportgen`.

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