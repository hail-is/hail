# Hail bull bull

[![Join the chat at https://gitter.im/broadinstitute/hail](https://badges.gitter.im/broadinstitute/hail.svg)](https://gitter.im/broadinstitute/hail?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge&utm_content=badge)

Hail is a framework for scalable genetic data analysis.  Hail is
pre-alpha software and under active development.  Hail is written in
Scala (mostly) and uses Apache [Spark](http://spark.apache.org/) and
other Apache Hadoop projects.  If you are interested in getting
involved in Hail development, email hail@broadinstitute.org.

## Documentation

Read the [docs](https://hail.is/docs/).

## Citing Hail

If you use Hail for published work, please cite both the software:

 - Hail, [https://github.com/broadinstitute/hail](https://github.com/broadinstitute/hail)

and the forthcoming manuscript describing Hail (if possible):

 - Cotton Seed, Alex Bloemendal, Jonathan M Bloom, Jacqueline I Goldstein, Daniel King, Timothy Poterba.  Hail: An Open-Source Framework for Scalable Genetic Data Analysis.  In preparation.

or the following paper which includes a brief introduction to Hail in the online methods:

 - Andrea Ganna, Giulio Genovese, Daniel P Howrigan, Andrea Byrnes, Mitja Kurki, Seyedeh M Zekavat, Christopher W Whelan, Robert E Handsaker, Mart Kals, Alex Bloemendal, Jonathan M Bloom, Jacqueline I Goldstein, Timothy Poterba, Cotton Seed, Michel G Nivard, Pradeep Natarajan, Reedik Magi, Diane Gage, Elise B Robinson, Andres Metspalu, Veikko Salomaa, Jaana Suvisaari, Shaun M Purcell, Pamela Sklar, Sekar Kathiresan, Mark J Daly, Steven A McCarroll, Patrick F Sullivan, Aarno Palotie, Tonu Esko, Christina Hultman, Benjamin M Neale. _Ultra-rare disruptive and damaging mutations influence educational attainment in the general population_.  doi: http://dx.doi.org/10.1101/050195.

## Roadmap

Here is a rough list of features currently planned or under
development:

 - generalized query language
 - better interoperability with other Hadoop projects
 - kinship estimation from GRM
 - LMM
 - burden tests, SKAT
 - logistic regression
 - posterior (PP)
 - LD pruning
 - TDT
 - Kaitlin Samocha's de novo caller
