[![Build Status](https://magnum.travis-ci.com/broadinstitute/hail.svg?token=BppUSW8Cb2YatFa34Fpx&branch=master)](https://magnum.travis-ci.com/broadinstitute/hail)

# Hail

## Table of Contents
1. [Importing](docs/Importing.md)
2. [QC](docs/QC.md)
3. [Filtering](docs/Filtering.md)
4. [Exporting to TSVs](docs/Exporting.md)

## Tools

We use the following development tools:
 - git
 - Scala
 - Gradle for build management
 - TestNG and [ScalaCheck](https://www.scalacheck.org/) for testing
 - Travis-CI for automated testing
 - Jacoco for code coverage
 - args4j for command line parsing
 - htsjdk for VCF parsing
 - Spark
 - Apache Commons libraries
 - IntelliJ (but you can use whatever editor you want)

For development, you only need to install git and gradle.  Gradle
handles the other dependencies.  On OSX, you can install gradle with
`brew install gradle`.

## Terminology

 - .vds: A .vds directory stores a `VariantDataset`, hail's internal
representation of the information in a .vcf file.  It is stored
(mostly) as parquet files.  You can use the hail `write` command to
create a .vds file from a a .vcf\[.bgz\] file.

## Building

To build hail, just do:

```
~/hail $ gradle installDist
```

This will populate `build/install` with an installation of hail.  Then
you can directly run `build/install/hail/bin/hail`.

## Running

To run the tests, do:

```
~/hail $ gradle check
```

To generate a code coverage report and view it, do:

```
~/hail $ gradle coverage
~/hail $ open build/build/reports/coverage/index.html
```

To convert a .vcf.gz to a .vds, do:

```
~/hail $ ./build/install/hail/bin/hail import -i src/test/resources/sample.vcf.gz write -o ~/sample.vds
```

`sample.vcf.gz` is a 182KB test `.vcf.gz` with a hundred or so samples
and variants.  This creates `~/sample.vds`.

To run sampleqc, do:

```
~/hail $ ./build/install/hail/bin/hail read -i ~/sample.vds sampleqc -o ~/sampleqc.tsv
```

For more options and commands, do `hail -h`.
