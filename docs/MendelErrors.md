# Mendel errors in Hail

The `mendelerrors` module finds all violations of Mendelian inheritance in each (dad, mom, kid) trio of samples.

Command line arguments:
 - `-f` -- a [Plink .fam file](https://www.cog-genomics.org/plink2/formats#fam)
 - `-o` -- a root name for output files

The command
```
mendelerrors -f trios.fam -o genomes
```
outputs four tsv files whose formats agree with [Plink](https://www.cog-genomics.org/plink2/formats#mendel):

- `genomes.mendel` -- all mendel errors (hadoop)
- `genomes.fmendel` -- error count per trio
- `genomes.imendel` -- error count per individual
- `genomes.lmendel` -- error count per locus (hadoop)

Each Mendel error is given a code, extending the [Plink classification](https://www.cog-genomics.org/plink2/basic_stats#mendel).
In the table below, ploidy of the kid is based on the pseudo-autosomal region (PAR) as follows:

- Auto -- autosome or PAR or female
- HemiX -- X and not PAR and male
- HemiY -- Y and not PAR and male

Dad    | Mom    | Kid    | Ploidy | Code
---    | ---    | ---    | ---    | ---
HomVar | HomVar |    Het | Auto   | 1
HomRef | HomRef |    Het | Auto   | 2
HomRef | Het/HomVar | HomVar | Auto   | 3
Het/HomVar | HomRef | HomVar | Auto   | 4
HomRef | HomRef | HomVar | Auto   | 5
HomVar | Het/HomVar | HomRef | Auto   | 6
Het/HomVar | HomVar | HomRef | Auto   | 7
HomVar | HomVar | HomRef | Auto   | 8
Any   | HomVar | HomRef | HemiX  | 9
Any   | HomRef | HomVar | HemiX  | 10
HomVar | Any   | HomRef | HemiY  | 11
HomRef | Any   | HomVar | HemiY  | 12

PAR is defined with respect to the reference [GRCh37](http://www.ncbi.nlm.nih.gov/projects/genome/assembly/grc/human/):

- X-60001:2699520
- X-154931044:155260560
- Y-10001:2649520
- Y-59034050:59363566

Mitochondrial DNA is ignored.
