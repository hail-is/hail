# Filtering in Hail

Hail includes three filtering modules:
 - `filtervariants`
 - `filtersamples`
 - `filtergenotypes`
  
The modules share much of their command-line interface, but there are some important differences.  Hail's modern filtering system is distinguished by the user's ability to evaluate a scala expression for each variant, sample, or genotype to determine whether to keep or remove those data.  This system is incredibly powerful, and allows for filtering procedures that might otherwise take multiple iterations or even multiple tools to be completed in one command.

Filtering commands have two required parameters, `-c` and `--keep/--remove`.  `-c` expects a string that refers to either a file path or boolean expression, and one of `--keep` or `--remove` is required.
  
## Using inclusion/exclusion files

1. `filtervariants` -- ".interval_list" file
 - Hail expects a .interval_list file to contain either three or five fields per line in the following formats: `contig  start  end` or `contig  start  end  direction  target`.  In either case, Hail will use only the `contig`, `start`, and `end` fields.  Each variant is evaluated against each line in the `.interval_list` file, and any match will mark the variant to be kept / excluded based on the presence of the `--keep` and `--remove` flags.  
 - _Note: "start" and "end" match positions inclusively, e.g. start <= position <= end_

2. `filtersamples` -- ".sample_list" file
 - Hail expects a .sample_list file to contain a newline-delimited list of sample ids.  The `--keep` and `--remove` command-line flags will determine whether the list of samples is excluded or kept.  
 - **As of 12/21/15, Hail does not allow users to specify mappings between the ID scheme used in import and other naming conventions.  This feature will be added in a future release.**  

3. `filtergenotypes` -- no inclusion/exclusion files supported

## Using expressions

Hail provides powerful utility in filtering by allowing users to write their own Scala boolean expressions on the command line, using the defined genotype, sample, variant, and annotation data.  This mode is used when the input to the `-c` command line argument does not match one of the expected inclusion/exclusion files extensions.

**Exposed namespaces:**
 - `filtersamples`: 
   - `s` (sample)
   - `sa` (sample annotation)
 - `filtervariants`:
   - `v` (variant)
   - `va` (variant annotation)
 - `filtergenotypes`:
   - `g` (genotype)
   - `s` (sample)
   - `sa` (sample annotation)
   - `v` (variant)
   - `va` (variant annotation)
  
These boolean expressions can be as simple or complicated as you like.  The below are all possible expressions:
```
filtervariants -c 'v.contig == "5"' --keep
```
```
filtervariants -c 'va.pass' --keep
```
```
filtervariants -c '!va.pass' --remove
```
```
filtersamples -c 'if ("DUMMY" ~ s.id) {sa.qc.rTiTv > .45 && sa.qc.nSingleton < 10000000} else sa.qc.rTiTv > .40' --keep
```
```
filtergenotypes -c 'g.gq < 20 || (g.gq < 30 && va.info.FST > 30)' --remove
```

**Supported comparisons and transformations:**
 - Numerical comparisons: `==`, `<`, `<=`, `>`, `>=`
 - Numerical transformations:
   - `.log(base = e)` -- log of the value
   
   - `.pwr(exponent)` -- raise the number to the given power
 - Array Operations:
   - apply: `arr(index)` -- get a value from the array, or NA if array or index is missing
   
**FIXME**


**Remember:**
 - All variables and values are case sensitive
 - Missing values will always be **excluded**, regardless of `--keep`/`--remove`
 - Always use logical and/or (`&&` / `||') instead of bitwise and/or (`&` / `|`).  The latter can cause inconsistencies **without throwing errors**
 - 
  
## Accessible fields of exposed classes

**Genotype:** `g`
 - `g.ad:          (Int, Int)` -- allelic depth for each allele
 - `g.dp:                 Int` -- the total number of informative reads
 - `g.isHomRef:       Boolean` -- true if this call is `0/0`
 - `g.isHet:          Boolean` -- true if this call is heterozygous
 - `g.isHomVar:       Boolean` -- true if this call is `1/1`
 - `g.isCalledNonRef: Boolean` -- true if either `g.isHet` or `g.isHomVar` is true
 - `g.isCalled:       Boolean` -- true if the genotype is `./.`
 - `g.isNotCalled:    Boolean` -- true if genotype is not `./.`
 - `g.gq:                 Int` -- the value of the lowest non-zero PL, or 0 if `./.`
 - `g.nNonRef:            Int` -- the number of called alternate alleles
 - `g.pAB:             Double` -- p-value for pulling the given allelic depth from a binomial distribution
 
**Variant:** `v`
 - `v.contig:          String` -- string representation of contig, exactly as imported.  _NB: Hail stores contigs as strings.  Use double-quotes when checking contig equality_
 - `v.start:              Int` -- SNP position or start of an indel
 - `v.ref:             String` -- reference allele
 - `v.alt:             String` -- Alternate allele
 - `v.wasSplit:       Boolean` -- true if this variant was originally multiallelic
 - `v.inParX:         Boolean` -- true if in pseudo-autosomal region on chromosome X
 - `v.inParY:         Boolean` -- true if in pseudo-autosomal region on chromosome Y
 - `v.isSNP:          Boolean` -- true if both `v.ref` and `v.alt` are single bases
 - `v.isMNP:          Boolean` -- true if `v.ref` and `v.alt` are the same (>1) length
 - `v.isIndel:        Boolean` -- true if `v.ref` and `v.alt` are not the same length
 - `v.isInsertion:    Boolean` -- true if `v.ref` is shorter than `v.alt`
 - `v.isDeletion:     Boolean` -- true if `v.ref` is longer than `v.alt`
 - `v.isComplex:      Boolean` -- true if `v` is not an indel, but `v.ref` and `v.alt` length do not match
 - `v.isTransition:   Boolean` -- true if the polymorphism is a purine-purine or pyrimidine-pyrimidine switch
 - `v.isTransversion: Boolean` -- true if the polymorphism is a purine-pyrimidine flip
 - `v.nMismatch:          Int` -- the total number of bases in `v.ref` and `v.alt` that do not match
 
**Sample:** `s`
 - `s.id:              String` The ID of this sample, as read at import-time
 
**Variant Annotations:** `va`

There are no mandatory methods for annotation classes.  Annotations are generated by hail modules, like the qc modules with the `--store` option.  However, there is an exception -- when a VCF file is imported, certain fields will be saved in annotations:
 - `va.pass:          Boolean` -- true if the variant contains `PASS` in the filter field (false if `.` or other)
 - `va.filters:   Set[String]` -- set containing the list of filters applied to a variant.  Accessible using `va.filters.contains("VQSRTranche99.5...")`, for example
 - `va.qual:           Double` -- the number in the qual field
 - `va.multiallelic:  Boolean` -- true if the variant is multiallelic or was split
 - `va.info.<field>:      Any` -- matches (with proper capitalization) any defined info field.  Data types match the definition in the vcf header, and if the `Number` is "A", "R", or "G", the result will be stored in an array.
 
If `variantqc --store` has been run:
 - `va.qc.<FIELD>:        Any` -- matches (with proper capitalization) variant qc fields.  [See list of available statistics here.](QC.md)

**Sample Annotations:** `sa`

There are no mandatory methods for annotation classes.  Annotations are generated by hail modules, like the qc modules with the `--store` option.  However, the following namespace will be available if `sampleqc --store` has been run:
 - `sa.qc.<FIELD>:        Any` -- matches (with proper capitalization) variant qc fields.  [See list of available statistics here.](QC.md)
