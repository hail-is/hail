[![Build Status](https://magnum.travis-ci.com/cseed/k3.svg?token=BppUSW8Cb2YatFa34Fpx&branch=master)](https://magnum.travis-ci.com/cseed/k3)

# k3

## Tools

We use the following development tools:
 - git
 - Scala
 - Gradle for build management
 - TestNG and [ScalaCheck](https://www.scalacheck.org/) for testing
 - Travis-CI for automated testing
 - Jacoco for code coverage
 - Spark
 - Apache Commons libraries
 - IntelliJ (but you can use whatever editor you want)

For development, you only need to install git and gradle.  Gradle
handles the other dependencies.  On OSX, you can install gradle with
`brew install gradle`.

## Terminology

 - .vds: A .vds directory stores a `VariantDataset`, k3's internal
representation of the information in a .vcf file.  It is stored
(mostly) as parquet files.  You can use the k3 `write` command to
create a .vds file from a a .vcf\[.gz\] file.

## Building

To build k3, just do:

```
~/k3 $ gradle installDist
```

This will populate `build/install` with an installation of k3.  Then
you can directly run `build/install/k3/bin/k3`.

## Running

To run the tests, do:

```
~/k3 $ gradle check
```

To generate a code coverage report and view it, do:

```
~/k3 $ gradle coverage
~/k3 $ open build/build/reports/coverage/index.html
```

To convert a .vcf.gz to a .vds, do:

```
~/k3 $ ./build/install/k3/bin/k3 src/test/resources/sample.vcf.gz write ~/sample.vds
```

`sample.vcf.gz` is a 182KB test `.vcf.gz` with a hundred or so samples
and variants.  This creates `~/sample.vcf`.

To run sampleqc, do:

```
~/k3 $ ./build/install/k3/bin/k3 ~/sample.vds sampleqc ~/sampleqc.tsv
```

For more options and commands, do `k3 -help`.
