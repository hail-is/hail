<div class="cmdhead"></div>

<div class="description"></div>

<div class="synopsis"></div>

<div class="options"></div>

<div class="cmdsubsection">

Hail supports basic family-based association testing for dichotomous traits, through the transmission disequilibrium test (TDT) [Spielman, McGinnis, and Ewens, 1993] (http://www.ncbi.nlm.nih.gov/pubmed/8447318).

The Hail TDT module requires a [Plink .fam file](https://www.cog-genomics.org/plink2/formats#fam) as input, and produces variant annotations.  The schema produced by this module is below:

```
Struct {
  nTransmitted: Int,
  nUntransmitted: Int,
  chiSquare: Double
}
```

In the above schema, `nTransmitted` is the total number of transmitted alternate alleles, `nUntransmitted` is the total number of untransmitted alternate alleles`, and `chiSquare` is the statistic derived from the probability of observing the given distribution under the null.  The TDT statistic is calculated simply as:

$(t-u)^2 \over (t+u)$

The Hail command to run a basic TDT analysis for family data:
```
tdt -f pedigree.fam --root va.tdt
```

where `t` and `u` are the number of transmitted and untransmitted alleles as shown in the T and U columns from `tdtres.tdt`; under the null, it is distributed as a 1 degree of freedom chi-squared statistic (see details).

#### Details
The TDT tracks the number of times the alternate allele is transmitted (t) or not transmitted (u) from a heterozgyous parent to an affected child under the null that the rate of such transmissions is 0.5.  In cases in which transmission is guaranteed (i.e., the Y chromosome, mitochondria, and paternal chromosome X variants outside of the PAR), the TDT cannot be used as it violates the null hypothesis.
The number of transmissions and untransmissions for each possible set of genotypes is determined from the table below.  The copy state of a locus with respect to a trio is defined as follows, where PAR is the pseudo-autosomal region (PAR).

- HemiX -- in non-PAR of X, male child
- Auto -- otherwise (in autosome or PAR, or female child)

Kid | Dad | Mom | Copy state | T | U
---|---|---|---|---|---
HomRef | Het | Het | Auto | 0 | 2
HomRef | HomRef | Het | Auto | 0 | 1
HomRef | Het | HomRef | Auto | 0 | 1
Het | Het | Het | Auto | 1 | 1
Het | HomRef | Het | Auto | 1 | 0
Het | Het | HomRef | Auto | 1 | 0
Het | HomVar | Het | Auto | 0 | 1
Het | Het | HomVar | Auto | 0 | 1
HomVar | Het | Het | Auto | 2 | 0
HomVar | Het | HomVar | Auto | 1 | 0
HomVar | HomVar | Het | Auto | 1 | 0
HomRef | HomRef | Het | HemiX | 0 | 1
HomRef | HomVar | Het | HemiX | 0 | 1
HomVar | HomRef | Het | HemiX | 1 | 0
HomVar | HomVar | Het | HemiX | 1 | 0

`tdt` only considers complete trios (two parents and a proband) with defined sex.

PAR is currently defined with respect to reference [GRCh37](http://www.ncbi.nlm.nih.gov/projects/genome/assembly/grc/human/):

- X: 60001-2699520
- X: 154931044-155260560
- Y: 10001-2649520
- Y: 59034050-59363566

`tdt` assumes all contigs apart from X and Y are fully autosomal; decoys, etc. are not given special treatment.
