# Mendel Errors in Hail

The `mendelerrors` module finds all Mendel errors in each (father, mother, child) trio of samples.

Command line arguments:
 - `-f` -- a .fam file
 - `-o` -- a root name for output files

The command
```
mendelerrors -f trios.fam -o genomes
```
outputs four tsv files:

- `genomes.mendel` -- all mendel errors (hadoop)
- `genomes.fmendel` -- error count per trio
- `genomes.imendel` -- error count per individual
- `genomes.lmendel` -- error count per locus (hadoop)

File formats agree with [Plink](https://www.cog-genomics.org/plink2/formats#mendel)

Dad    | Mom    | Kid    | Ploidy | Code
---    | ---    | ---    | ---    | ---
HomVar | HomVar |    Het | Auto   | 1
HomRef | HomRef |    Het | Auto   | 2
HomRef | !HomRef | HomVar | Auto   | 3
!HomRef | HomRef | HomVar | Auto   | 4
HomRef | HomRef | HomVar | Auto   | 5
HomVar | !HomRef | HomRef | Auto   | 6
!HomREf | HomVar | HomRef | Auto   | 7
HomVar | HomVar | HomRef | Auto   | 8
      | HomVar | HomRef | HemiX  | 9
      | HomRef | HomVar | HemiX  | 10
HomVar |       | HomRef | HemiY  | 11
HomRef |       | HomVar | HemiY  | 12

Test.