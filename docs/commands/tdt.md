<div class="cmdhead"></div>

<div class="description"></div>

<div class="synopsis"></div>

<div class="options"></div>

<div class="cmdsubsection">

Hail supports family-based association testing for dichotomous traits through the transmission disequilibrium test (TDT) described in [Spielman, McGinnis, and Ewens, 1993](http://www.ncbi.nlm.nih.gov/pubmed/8447318).

The Hail TDT module requires a [Plink .fam file](https://www.cog-genomics.org/plink2/formats#fam) as input.  The command
```
tdt -f pedigree.fam
```
produces four variant annotations:

Annotation | Type | Value
---|---|---
`va.tdt.nTransmitted` | Int | number of transmitted alternate alleles
`va.tdt.nUntransmitted` | Int | number of untransmitted alternate alleles
`va.tdt.chi2` | Double | TDT statistic
`va.tdt.pval` | Double | $p$-value

#### Details
The transmission disequilibrium test tracks the number of times the alternate allele is transmitted (t) or not transmitted (u) from a heterozgyous parent to an affected child under the null that the rate of such transmissions is $0.5$.  In cases in which transmission is guaranteed (i.e., the Y chromosome, mitochondria, and paternal chromosome X variants outside of the PAR), the test cannot be used.

The TDT statistic is given by

$$(t-u)^2 \over (t+u)$$

and follows a 1 d.o.f. chi-squared distribution under the null hypothesis.


The number of transmissions and untransmissions for each possible set of genotypes is determined from the table below.  The copy state of a locus with respect to a trio is defined as follows, where PAR is the pseudo-autosomal region (PAR).

- HemiX -- in non-PAR of X and child is male
- Auto -- otherwise (in autosome or PAR, or child is female)

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
