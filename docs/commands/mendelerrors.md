<div class="cmdhead"></div>

<div class="description"></div>

<div class="synopsis"></div>

<div class="options"></div>

<div class="cmdsubsection">
### Implementation Details:

The `mendelerrors` command finds all violations of Mendelian inheritance in each (dad, mom, kid) trio of samples.

The command
```
mendelerrors -f trios.fam -o genomes
```
outputs four TSV files according to the [Plink mendel formats](https://www.cog-genomics.org/plink2/formats#mendel):

- `genomes.mendel` -- all mendel errors: FID KID CHR SNP CODE ERROR
- `genomes.fmendel` -- error count per nuclear family: FID PAT MAT CHLD N NSNP
- `genomes.imendel` -- error count per individual: FID IID N NSNP
- `genomes.lmendel` -- error count per variant: CHR SNP N

FID, KID, PAT, MAT, and IID refer to family, kid, dad, mom, and individual ID, respectively, with missing values set to `0`.
SNP denotes the variant identifier `chr:pos:ref:alt`.
N counts all errors, while NSNP only counts SNP errors (NSNP is not in Plink).
CHLD is the number of children in a nuclear family.

The CODE of each Mendel error is determined by the table below, extending the [Plink classification](https://www.cog-genomics.org/plink2/basic_stats#mendel).
Those individuals implicated by each code are in bold.
The copy state of a locus with respect to a trio is defined as follows, where PAR is the pseudo-autosomal region (PAR).

- HemiX -- in non-PAR of X, male child
- HemiY -- in non-PAR of Y, male child
- Auto -- otherwise (in autosome or PAR, or female child)

Any refers to {HomRef, Het, HomVar, NoCall} and ! denotes complement in this set.

Code | Dad | Mom | Kid | Copy state
---|---|---|---|---
1 | **HomVar** | **HomVar** | **Het** | Auto
2 | **HomRef** | **HomRef** |  **Het** | Auto
3 | **HomRef** | ! HomRef | **HomVar** | Auto
4 | ! HomRef | **HomRef** | **HomVar** | Auto
5 | HomRef | HomRef | **HomVar** | Auto
6 | **HomVar** | ! HomVar | **HomRef** | Auto
7 | ! HomVar | **HomVar** | **HomRef** | Auto
8 | HomVar | HomVar | **HomRef** | Auto
9 | Any | **HomVar** | **HomRef** | HemiX
10 | Any | **HomRef** | **HomVar** | HemiX
11 | **HomVar** | Any | **HomRef** | HemiY
12 | **HomRef** | Any | **HomVar** | HemiY

</div>

<div class="cmdsubsection">
### Notes:

`mendelerrors` only considers children with two parents and defined sex.

PAR is currently defined with respect to reference [GRCh37](http://www.ncbi.nlm.nih.gov/projects/genome/assembly/grc/human/):

- X: 60001-2699520
- X: 154931044-155260560
- Y: 10001-2649520
- Y: 59034050-59363566

`mendelerrors` assumes all contigs apart from X and Y are fully autosomal; mitochondria, decoys, etc. are not given special treatment.

</div>