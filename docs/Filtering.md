# Filtering

Hail includes three filtering modules:
 - `filtervariants`
 - `filtersamples`
 - `filtergenotypes`
  
The modules share much of their command-line interface, but there are some important differences.  Hail's modern filtering system is distinguished by the user's ability to evaluate a scala expression for each variant, sample, or genotype to determine whether to keep or remove those data.  This system is incredibly powerful, and allows for filtering procedures that might otherwise take multiple iterations or even multiple tools to be completed in one command.

Command line arguments: 
 - `-c | --condition <cond>` -- filter expression (see below) or path to file of appropriate type
 - `--keep/--remove` -- determines behavior of file interval/list or boolean expression
  
## Using inclusion/exclusion files

1. `filtervariants` -- ".interval_list" file
 - Hail expects a .interval_list file to contain either three or five fields per line in the following formats: `contig:start-end` or `contig  start  end  direction  target` (TSV).  In either case, Hail will use only the `contig`, `start`, and `end` fields.  Each variant is evaluated against each line in the `.interval_list` file, and any match will mark the variant to be kept / excluded based on the presence of the `--keep` and `--remove` flags.  
 - _Note: "start" and "end" match positions inclusively, e.g. start <= position <= end_

2. `filtersamples` -- ".sample_list" file
 - Hail expects a .sample_list file to contain a newline-delimited list of sample ids.  The `--keep` and `--remove` command-line flags will determine whether the list of samples is excluded or kept.  This file can contain sample IDs not present in the VDS.  

3. `filtergenotypes` -- no inclusion/exclusion files supported

## Using expressions

Hail provides powerful utility in filtering by allowing users to write their own boolean expressions on the command line, using the exposed genotype, sample, variant, and annotation objects.  This mode is used when the input to the `-c` command line argument does not match one of the expected inclusion/exclusion files extensions.

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
filtergenotypes -c 'g.ad(1).toDouble / g.dp < 0.2' --remove
```
```
filtersamples -c 'if ("DUMMY" ~ s.id) {sa.qc.rTiTv > .45 && sa.qc.nSingleton < 10000000} else sa.qc.rTiTv > .40' --keep
```
```
filtergenotypes -c 'g.gq < 20 || (g.gq < 30 && va.info.FS > 30)' --remove
```

## Accessible fields of exposed classes

**Genotype:** `g`
 - `g.gt                  Int` -- the call, `gt = k*(k+1)/2+j` for call `j/k`
 - `g.ad:          Array[Int]` -- allelic depth for each allele
 - `g.dp:                 Int` -- the total number of informative reads
 - `g.od                  Int` -- `od = dp - ad.sum`
 - `g.gq:                 Int` -- the difference between the two smallest PL entries
 - `g.isHomRef:       Boolean` -- true if this call is `0/0`
 - `g.isHet:          Boolean` -- true if this call is heterozygous
 - `g.isHetRef:       Boolean` -- true if this call is `0/k` with `k>0`
 - `g.isHetNonRef:    Boolean` -- true if this call is `j/k` with `j>0`
 - `g.isHomVar:       Boolean` -- true if this call is `j/j` with `j>0`
 - `g.isCalledNonRef: Boolean` -- true if either `g.isHet` or `g.isHomVar` is true
 - `g.isCalled:       Boolean` -- true if the genotype is not `./.`
 - `g.isNotCalled:    Boolean` -- true if the genotype is `./.`
 - `g.nNonRef:            Int` -- the number of called alternate alleles
 - `g.pAB():           Double` -- p-value for pulling the given allelic depth from a binomial distribution with mean 0.5.  Assumes the variant `v` is biallelic.
 
**Variant:** `v`
 - `v.contig:                String` -- string representation of contig, exactly as imported.  _NB: Hail stores contigs as strings.  Use double-quotes when checking contig equality_
 - `v.start:                    Int` -- SNP position or start of an indel
 - `v.ref:                   String` -- reference allele sequence
 - `v.isBiallelic:          Boolean` -- true if `v` is biallelic
 - `v.nAlleles:                 Int` -- number of alleles
 - `v.nAltAlleles:              Int` -- number of alternate alleles, equal to `nAlleles - 1`
 - `v.nGenotypes:               Int` -- number of genotypes
 - `v.altAlleles: Array[AltAlleles]` -- the alternate alleles
 - `v.inParX:               Boolean` -- true if in pseudo-autosomal region on chromosome X
 - `v.inParY:               Boolean` -- true if in pseudo-autosomal region on chromosome Y
 - `v.altAllele:          AltAllele` -- The Alternate allele.  Assumes biallelic.
 - `v.alt:                   String` -- Alternate allele sequence.  Assumes biallelic.

**AltAllele:**
 - `aa.ref:             String` -- reference allele sequence
 - `aa.alt:             String` -- alternate allele sequence
 - `aa.isSNP:          Boolean` -- true if both `v.ref` and `v.alt` are single bases
 - `aa.isMNP:          Boolean` -- true if `v.ref` and `v.alt` are the same (>1) length
 - `aa.isIndel:        Boolean` -- true if `v.ref` and `v.alt` are not the same length
 - `aa.isInsertion:    Boolean` -- true if `v.ref` is shorter than `v.alt`
 - `aa.isDeletion:     Boolean` -- true if `v.ref` is longer than `v.alt`
 - `aa.isComplex:      Boolean` -- true if `v` is not an indel, but `v.ref` and `v.alt` length do not match
 - `aa.isTransition:   Boolean` -- true if the polymorphism is a purine-purine or pyrimidine-pyrimidine switch
 - `aa.isTransversion: Boolean` -- true if the polymorphism is a purine-pyrimidine flip
 - `aa.nMismatch:          Int` -- the total number of bases in `v.ref` and `v.alt` that do not match
 
**Sample:** `s`
 - `s.id:              String` The ID of this sample, as read at import-time
 
**Variant Annotations:** `va`

There are no mandatory methods for annotation classes.  Annotations are generated by hail modules, like the qc modules with the `--store` option.  However, there is an exception -- when a VCF file is imported, certain fields will be saved in annotations:
 - `va.pass:          Boolean` -- true if the variant contains `PASS` in the filter field (false if `.` or other)
 - `va.filters:   Set[String]` -- set containing the list of filters applied to a variant.  Accessible using `va.filters.contains("VQSRTranche99.5...")`, for example
 - `va.rsid:           String` -- rsid of the variant, if it has one ("." otherwise)
 - `va.qual:           Double` -- the number in the qual field
 - `va.multiallelic:  Boolean` -- true if the variant is multiallelic or was split
 - `va.info.<field>:      Any` -- matches (with proper capitalization) any defined info field.  Data types match the definition in the vcf header, and if the `Number` is "A", "R", or "G", the result will be stored in an array.
If `splitmulti` has been run:
 - `va.wasSplit       Boolean` -- this variant was originally multiallelic 
 - `va.aIndex             Int` -- The original index of this variant in imported line, 0 for imported biallelic sites
If `variantqc --store` has been run:
 - `va.qc.<FIELD>:        Any` -- matches (with proper capitalization) variant qc fields.  [See list of available computed statistics here.](QC.md)

**Sample Annotations:** `sa`

There are no mandatory methods for annotation classes.  Annotations are generated by hail modules, like the qc modules with the `--store` option.  However, the following namespace will be available if `sampleqc --store` has been run:
 - `sa.qc.<FIELD>:        Any` -- matches (with proper capitalization) variant qc fields.  [See list of available computed statistics here.](QC.md)
