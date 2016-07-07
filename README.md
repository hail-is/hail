# Hail

Hail is a framework for scalable genetic data analysis.  Hail is
pre-alpha software and under active development.  Hail is written in
Scala (mostly) and uses Apache [Spark](http://spark.apache.org/) and
other Apache Hadoop projects.  If you are interested in getting
involved in Hail development, email hail@broadinstitute.org.

## Documentation

 - [Building](docs/Building.md)
 - [Representation](docs/Representation.md)
 - [Hail's expression language](docs/HailExpressionLanguage.md)
 - [Importing](docs/Importing.md)
 - [Splitting Multiallelic Variants](docs/Splitmulti.md)
 - [Renaming Samples](docs/RenameSamples.md)
 - [Annotating Samples or Variants](docs/ImportAnnotations.md)
 - [Quality Control](docs/QC.md)
 - [PCA](docs/PCA.md)
 - [Annotating with the Variant Effect Predictor](docs/VEP.md)
 - [Filtering](docs/Filtering.md)
 - [Linear regression](docs/LinearRegression.md)
 - [Mendel errors](docs/MendelErrors.md)
 - [Exporting to TSV](docs/ExportTSV.md)
 - [Exporting to VCF](docs/ExportVCF.md)
 - [Exporting to Plink](docs/ExportPlink.md)
 - [Persist](docs/Persist.md)

## Roadmap

Here is a rough list of features currently planned or under
development:

 - generalized query language
 - better interoperability with other Hadoop projects
 - kinship estimation from GRM
 - LMM
 - burden tests, SKAT
 - logistic regression
 - dosage
 - posterior (PP)
 - LD pruning
 - sex check
 - TDT
 - BGEN
 - Kaitlin Samocha's de novo caller
