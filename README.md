# Hail

## Table of Contents

1. [Representation](docs/Representation.md)
2. [Importing](docs/Importing.md)
3. [Splitting Multiallelic Variants](docs/Splitmulti.md)
4. [QC](docs/QC.md)
5. [Filtering](docs/Filtering.md)
6. [Renaming Samples](docs/RenameSamples.md)
6. [Exporting to TSVs](docs/ExportTSV.md)
8. [Exporting to Plink](docs/ExportPlink.md)
7. [Exporting to VCF](docs/ExportVCF.md)
9. [Mendel errors](docs/MendelErrors.md)

## Tools

We use the following development tools:
 - git
 - Scala
 - Gradle for build management
 - TestNG for testing
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
~/hail $ ./build/install/hail/bin/hail importvcf src/test/resources/sample.vcf.gz write -o ~/sample.vds
```

`sample.vcf.gz` is a 182KB test `.vcf.gz` with a hundred or so samples
and variants.  This creates `~/sample.vds`.

To run sampleqc, do:

```
~/hail $ ./build/install/hail/bin/hail read -i ~/sample.vds sampleqc -o ~/sampleqc.tsv
```

For more options and commands, do `hail -h`.
