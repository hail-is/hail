<div class="cmdhead"></div>

<div class="description"></div>

<div class="synopsis"></div>

<div class="options"></div>

<div class="cmdsubsection">
### Notes:

 - Hail is designed to be maximally compatible with files in the [VCF v4.2 spec](https://samtools.github.io/hts-specs/VCFv4.2.pdf).

 - `importvcf` takes a list of VCF files to load.  All files must have the same header and the same set of samples in the same order (e.g., a dataset split by chromosome).  Files can be specified as [Hadoop glob patterns](#hadoopglob).
 
 - Ensure that the VCF file is correctly prepared for import:
   - VCFs should be either uncompressed (".vcf") or block-compressed (".vcf.bgz").  If you have a large compressed VCF that ends in ".vcf.gz", it is likely that the file is actually block-compressed, and you should rename the file to ".vcf.bgz" accordingly.  If you actually have a standard gzipped file, it is possible to import it to hail using the `-f` option.  However, this is not recommended -- all parsing will have to take place on one node, because gzip decompression is not parallelizable.  In this case, import could take significantly magnitude longer.
   - VCFs should reside in the hadoop file system
 - Run a hail command with `importvcf`.  The below command will read a .vcf.bgz file and write to a .vds file (Hail's preferred format).  It is possible to import and operate directly on a VCF file without first doing an import/write step, but this will greatly increase compute time if multiple commands are run (it is significantly faster to read a vds than import a vcf).
``` 
$ hail importvcf /path/to/file.vcf.bgz write -o /path/to/output.vds
```

 - Hail makes certain assumptions about the genotype fields, see [Representation](#Representation).  On import, Hail filters (sets to no-call) any genotype that violates these assumptions.  Hail interpets the format fields: GT, AD, OD, DP, GQ, PL; all others are silently dropped.

</div>


<div class="cmdsubsection">
### <a name="importvcf_annotations"></a>Annotations:

 - `va.pass:          Boolean` -- true if the variant contains `PASS` in the filter field (false if `.` or other)
 - `va.filters:   Set[String]` -- set containing the list of filters applied to a variant.  Accessible using `va.filters.contains("VQSRTranche99.5...")`, for example
 - `va.rsid:           String` -- rsid of the variant, if it has one ("." otherwise)
 - `va.qual:           Double` -- the number in the qual field
 - `va.info.<field>:        T` -- matches (with proper capitalization) any defined info field.  Data types match the type specified in the vcf header, and if the `Number` is "A", "R", or "G", the result will be stored in an array (accessed with array\[index\]).
</div>