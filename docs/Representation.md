
# Representation of sequence data in Hail

```
                Columns keyed
                by Samples  -->
            __________________________    __
           |                          |  |  |
           |                          |  |  |
   Rows    |    __                    |  |  |   Variant
 keyed by  |   |__| Genotype          |  |  | Annotations
 Variants  |                          |  |  |
           |                          |  |  |
    |      |                          |  |  |
    |      |                          |  |  |
    V      |                          |  |  |
           |                          |  |  |
           |                          |  |  |
           |                          |  |  |
           |__________________________|  |__|
            __________________________
           |__________________________|
                Sample annotations
           
                     ____
                    |    | Global annotations
                    |____|           
```

The above cartoon depicts the rough organization of the data stored in Hail.  The majority of the data is genotype information, which can be thought of as a matrix with columns keyed by [**sample**](#sample) objects, and rows keyed by [**variant**](#variant) objects.  Each cell of the matrix is a [**genotype**](#genotype) object.

For more information about **annotations**, [see the documentation here](Annotations.md)

Hail's internal representation for genotypes is conceptually similar to VCF.  Hail only supports diploid genotypes and does not yet store phasing information.  Hail uses a fixed set of genotype fields corresponding to a VCF format field of the form:
```
GT:AD:DP:GQ:PL
```

In addition, Hail considers OD = DP - sum(AD).

Hail makes the following assumptions about the genotype fields:
 - if both are present, PL(GT) = 0
 - GQ is present if and only if PL is present
 - GQ is the difference of the two smallest PL entries
 - if OD is present, then AD is present
 - if they are all present, sum(AD) + OD = DP
 - sum(AD) <= DP

Internally, Hail preserves these invariants.  On import, Hail filters (sets to no-call) any genotype that violates these invariants and prints a warning message.

## Properties and methods of Hail objects
 
<a name="variant"></a>
**Variant:** `v`

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
`v.inParX`              |  `Boolean`  | true if in pseudo-autosomal region on chromosome X
`v.inParY`              |  `Boolean`  | true if in pseudo-autosomal region on chromosome Y
`v.altAllele`           | `AltAllele` | The alternate allele (schema below).  **Assumes biallelic.**
`v.alt`                 | `String`    | Alternate allele sequence.  **Assumes biallelic.**

**AltAllele:** `v.altAlleles[idx]` or `v.altAllele` (biallelic) 

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
 `<altAllele>.nMismatch`      | `Int`     | the total number of bases in `v.ref` and `v.alt` that do not match

____ 
 
<a name="sample"></a>
**Sample:** `s`

Identifier | Type | Description
--- | :-: | ---
`s.id` | `String` | The ID of this sample, as read at import-time

____

<a name="genotype"></a>
**Genotype:** `g`

Identifier | Type | Description
--- | :-: | ---
`g.gt`             | `Int`     | the call, `gt = k*(k+1)/2+j` for call `j/k`
`g.ad`          | `Array[Int]` | allelic depth for each allele
`g.dp`             | `Int`     | the total number of informative reads
`g.od`             | `Int`     | `od = dp - ad.sum`
`g.gq`             | `Int`     | the difference between the two smallest PL entries
`g.pl`          | `Array[Int]` | phred-scaled normalized genotype likelihood values
`g.isHomRef`       | `Boolean` | true if this call is `0/0`
`g.isHet`          | `Boolean` | true if this call is heterozygous
`g.isHetRef`       | `Boolean` | true if this call is `0/k` with `k>0`
`g.isHetNonRef`    | `Boolean` | true if this call is `j/k` with `j>0`
`g.isHomVar`       | `Boolean` | true if this call is `j/j` with `j>0`
`g.isCalledNonRef` | `Boolean` | true if either `g.isHet` or `g.isHomVar` is true
`g.isCalled`       | `Boolean` | true if the genotype is not `./.`
`g.isNotCalled`    | `Boolean` | true if the genotype is `./.`
`g.nNonRefAlleles`        | `Int`     | the number of called alternate alleles
`g.pAB()`          | `Double`  | p-value for pulling the given allelic depth from a binomial distribution with mean 0.5.  Assumes the variant `v` is biallelic.
`g.fractionReadsRef` | `Double` | the ratio of ref reads to the sum of all *informative* reads
`g.fakeRef`        | `Boolean` | true if this genotype was downcoded in `splitmulti`.  This can happen if a 1/2 call is split to 0/1, 0/1
 