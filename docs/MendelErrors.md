# Mendel errors in Hail

The `mendelerrors` module finds all violations of Mendelian inheritance in each (dad, mom, kid) trio of samples.

Command line arguments:
 - `-f | --fam <filename>` -- a [Plink .fam file](https://www.cog-genomics.org/plink2/formats#fam)
 - `-o | --output <fileroot>` -- a root name for output files

*Note: currently all samples in the .fam file must have defined sex and exist in the variant data set.*

The command
```
mendelerrors -f trios.fam -o genomes
```
outputs four tsv files according to the [Plink mendel formats](https://www.cog-genomics.org/plink2/formats#mendel):

- `genomes.mendel` -- all mendel errors: FID KID CHR SNP CODE ERROR (hadoop)
- `genomes.fmendel` -- error count per nuclear family: FID PAT MAT CHLD N NSNP
- `genomes.imendel` -- error count per individual: FID IID N NSNP
- `genomes.lmendel` -- error count per variant: CHR SNP N (hadoop)

FID, KID, PAT, MAT, and IID refer to family, kid, dad, mom, and individual ID, respectively, with missing values set to `0`.
SNP denotes the variant identifier `chr:pos:ref:alt`.
N counts all errors, while NSNP only counts SNP errors (NSNP is not in Plink).
CHLD is the number of children in a nuclear family.

Each Mendel error is given a CODE as defined in table below, extending the [Plink classification](https://www.cog-genomics.org/plink2/basic_stats#mendel).
Those individuals implicated by each code are in bold.
Ploidy is based on the pseudo-autosomal region (PAR):

- HemiX -- non-PAR X in male child
- HemiY -- non-PAR Y in male child
- Auto -- otherwise (autosome or PAR or female child)

Any refers to {HomRef, Het, HomVar, NoCall} and ! denotes complement in this set.

Code | Dad | Mom | Kid | Ploidy
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

PAR is currently defined with respect to reference [GRCh37](http://www.ncbi.nlm.nih.gov/projects/genome/assembly/grc/human/):

- X: 60001-2699520
- X: 154931044-155260560
- Y: 10001-2649520
- Y: 59034050-59363566

`mendelerrors` assumes all contigs apart from X and Y are fully autosomal; mitochondrial DNA is not currenly supported.