# Hail Object Methods
 
## <a class="jumptarget" name="variant"></a>Variant

**Variable Name:** `v`

Identifier | Type | Description
--- | :-: | ---
`v.contig`              | `String`    | string representation of contig, exactly as imported.  _NB: Hail stores contigs as strings.  Use double-quotes when checking contig equality_
`v.start`               | `Int`       | SNP position or start of an indel
`v.ref`                 | `String`    | reference allele sequence
`v.isBiallelic`         | `Boolean`   | true if `v` has one alternate allele
`v.nAlleles`            | `Int`       | number of alleles
`v.nAltAlleles`         | `Int`       | number of alternate alleles, equal to `nAlleles - 1`
`v.nGenotypes`          | `Int`       | number of genotypes
`v.altAlleles`  | `Array[AltAllele]`  | the alternate alleles
`v.inXPar`              |  `Boolean`  | true if chromosome is X and start is in pseudo-autosomal region of X
`v.inYPar`              |  `Boolean`  | true if chromosome is Y and start is in pseudo-autosomal region of Y. _NB: most callers assign variants in PAR to X_
`v.inXNonPar`           |  `Boolean`  | true if chromosome is X and start is not in pseudo-autosomal region of X
`v.inYNonPar`           |  `Boolean`  | true if chromosome is Y and start is not in pseudo-autosomal region of Y
`v.altAllele`           | `AltAllele` | The alternate allele (schema below).  **Assumes biallelic.**
`v.alt`                 | `String`    | Alternate allele sequence.  **Assumes biallelic.**
`v.locus`               | `Locus`     | Chromosomal locus (chr, pos) of this variant
`v.isAutosomal`         | `Boolean`   | true if chromosome is not X, not Y, and not MT

## <a class="jumptarget" name="altallele"></a>AltAllele 

**Variable Name:** `v.altAlleles[idx]` or `v.altAllele` (biallelic)

Identifier | Type | Description
--- | :-: | ---
 `<altAllele>.ref`            | `String`  | reference allele sequence
 `<altAllele>.alt`            | `String`  | alternate allele sequence
 `<altAllele>.isSNP`          | `Boolean` | true if both `v.ref` and `v.alt` are single bases
 `<altAllele>.isMNP`          | `Boolean` | true if `v.ref` and `v.alt` are the same (>1) length
 `<altAllele>.isIndel`        | `Boolean` | true if `v.ref` and `v.alt` are not the same length
 `<altAllele>.isInsertion`    | `Boolean` | true if `v.ref` is shorter than `v.alt`
 `<altAllele>.isDeletion`     | `Boolean` | true if `v.ref` is longer than `v.alt`
 `<altAllele>.isComplex`      | `Boolean` | true if `v` is not an indel, but `v.ref` and `v.alt` length do not match
 `<altAllele>.isTransition`   | `Boolean` | true if the polymorphism is a purine-purine or pyrimidine-pyrimidine switch
 `<altAllele>.isTransversion` | `Boolean` | true if the polymorphism is a purine-pyrimidine flip

## <a class="jumptarget" name="locus"></a>Locus

**Variable Name:** `v.locus` or `Locus(chr, pos)`

Identifier | Type | Description
--- | :-: | ---
`<locus>.contig`   |  `String` |  String representation of contig
`<locus>.position` |  `Int`    |  Chromosomal position

## <a class="jumptarget" name="interval"></a>Interval

**Variable Name:** `Interval(locus1, locus2)`

Identifier | Type | Description
--- | :-: | ---
`<interval>.start` |  `Locus` | `Locus` object (see above) at the start of the interval (inclusive)
`<interval>.end`   |  `Locus` | `Locus` object (see above) at the end of the interval (exclusive)


## <a class="jumptarget" name="sample"></a>Sample
 
**Variable Name:** `s`

Identifier | Type | Description
--- | :-: | ---
`s.id` | `String` | The ID of this sample, as read at import-time


## <a class="jumptarget" name="genotype"></a>Genotype

**Variable Name:** `g`

Identifier | Type | Description
--- | :-: | ---
`g.gt`             | `Int`     | the call, `gt = k*(k+1)/2+j` for call `j/k`
`g.ad`          | `Array[Int]` | allelic depth for each allele
`g.dp`             | `Int`     | the total number of informative reads
`g.od`             | `Int`     | `od = dp - ad.sum`
`g.gq`             | `Int`     | the difference between the two smallest PL entries
`g.pl`          | `Array[Int]` | phred-scaled normalized genotype likelihood values
`g.dosage`   | `Array[Double]` | the linear-scaled probabilities
`g.isHomRef`       | `Boolean` | true if this call is `0/0`
`g.isHet`          | `Boolean` | true if this call is heterozygous
`g.isHetRef`       | `Boolean` | true if this call is `0/k` with `k>0`
`g.isHetNonRef`    | `Boolean` | true if this call is `j/k` with `j>0`
`g.isHomVar`       | `Boolean` | true if this call is `j/j` with `j>0`
`g.isCalledNonRef` | `Boolean` | true if either `g.isHet` or `g.isHomVar` is true
`g.isCalled`       | `Boolean` | true if the genotype is not `./.`
`g.isNotCalled`    | `Boolean` | true if the genotype is `./.`
`g.nNonRefAlleles`        | `Int`     | the number of called alternate alleles
`g.pAB`          | `Double`  | p-value for pulling the given allelic depth from a binomial distribution with mean 0.5.  Assumes the variant `v` is biallelic.
`g.fractionReadsRef` | `Double` | the ratio of ref reads to the sum of all *informative* reads
`g.fakeRef`        | `Boolean` | true if this genotype was downcoded in [`splitmulti`](#splitmulti).  This can happen if a `1/2` call is split to `0/1`, `0/1`
`g.isDosage` |`Boolean` | true if the data was imported from `importgen` or `importbgen`
`g.oneHotAlleles(Variant)` | `Array[Int]` | Produces an array of called counts for each allele in the variant (including reference).  For example, calling this function with a biallelic variant on hom-ref, het, and hom-var genotypes will produce `[2, 0]`, `[1, 1]`, and `[0, 2]` respectively.
`g.oneHotGenotype(Variant)` | `Array[Int]` | Produces an array with one element for each possible genotype in the variant, where the called genotype is 1 and all else 0.  For example, calling this function with a biallelic variant on hom-ref, het, and hom-var genotypes will produce `[1, 0, 0]`, `[0, 1, 0]`, and `[0, 0, 1]` respectively.
`g.gtj`             | `Int`     | the index of allele `j` for call `j/k` (0 = ref, 1 = first alt allele, etc.)
`g.gtk`             | `Int`     | the index of allele `k` for call `j/k` (0 = ref, 1 = first alt allele, etc.)


The conversion between `g.pl` (Phred-scaled likelihoods) and `g.dosage` (linear-scaled probabilities) assumes a uniform prior.
