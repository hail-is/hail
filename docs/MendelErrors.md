# Mendel errors in Hail

The `mendelerrors` module finds all violations of Mendelian inheritance in each (dad, mom, kid) trio of samples.

Command line arguments:
 - `-f` -- a [Plink .fam file](https://www.cog-genomics.org/plink2/formats#fam)
 - `-o` -- a root name for output files

The command
```
mendelerrors -f trios.fam -o genomes
```
outputs four tsv files which currently adhere to the [Plink mendel formats](https://www.cog-genomics.org/plink2/formats#mendel):

- `genomes.mendel` -- all errors: FID KID CHR SNP CODE ERROR (hadoop)
- `genomes.fmendel` -- count per nuclear family: FID PAT MAT CHLD N
- `genomes.imendel` -- count per individual: FID IID N
- `genomes.lmendel` -- count per locus: CHR SNP N (hadoop)

FID, KID, PAT, MAT, and IID refer to family, kid, dad, mom, and individual ID, respectively (missing values are `0`).
SNP is variant start position. CHLD is the number of children in a nuclear family.
In the table below, ploidy of the kid is based on the pseudo-autosomal region (PAR):
- Auto -- autosome or PAR or female
- HemiX -- X and not PAR and male
- HemiY -- Y and not PAR and male
Each Mendel error is given a CODE, extending the [Plink classification](https://www.cog-genomics.org/plink2/basic_stats#mendel). For each code, those individuals implicated in the error are in bold.

Code | Dad | Mom | Kid | Ploidy
1 | **HomVar** | **HomVar** | **Het** | Auto
2 | **HomRef** | **HomRef** |  **Het** | Auto
3 | **HomRef** | Het/HomVar | **HomVar** | Auto
4 | Het/HomVar | **HomRef** | **HomVar** | Auto
5 | **HomRef** | **HomRef** | **HomVar** | Auto
6 | **HomVar** | HomRef/Het | **HomRef** | Auto
7 | HomRef/Het | **HomVar** | **HomRef** | Auto
8 | **HomVar** | **HomVar** | **HomRef** | Auto
9 | Any | **HomVar** | **HomRef** | HemiX
10 | Any | **HomRef** | **HomVar** | HemiX
11 | **HomVar** | Any | **HomRef** | HemiY
12 | **HomRef** | Any | **HomVar** | HemiY

PAR is defined with respect to the reference [GRCh37](http://www.ncbi.nlm.nih.gov/projects/genome/assembly/grc/human/):

- X-60001:2699520
- X-154931044:155260560
- Y-10001:2649520
- Y-59034050:59363566

Mitochondrial DNA is ignored. Each trio is treated independently. So a quad consists of two trios. Nuclear family defined by parents.