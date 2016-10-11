

# Getting started

All you'll need is the [Java 8 JDK](http://www.oracle.com/technetwork/java/javase/downloads/index.html) and the Hail source code. To clone the [Hail repository](https://github.com/broadinstitute/hail) using [Git](https://git-scm.com/), run
```
$ git clone https://github.com/broadinstitute/hail.git
$ cd hail
```
The following commands are relative to the `hail` directory.

You can also download the source code directly from [here](https://github.com/broadinstitute/hail/archive/master.zip).

## Building and running Hail

Hail may be built to run locally or on a Spark cluster. Running locally is useful for getting started, analyzing or experimenting with small datasets, and Hail development.

### Locally
The single command
```
$ ./gradlew installDist
```
installs Hail at `build/install/hail`. The initial build takes time as [Gradle](https://gradle.org/) installs all Hail dependencies. The executable is `build/install/hail/bin/hail` (to run using `hail` add `build/install/hail/bin` to your path).

Here are a few simple things to try. To list all commands, run
```
$ ./build/install/hail/bin/hail -h
```
To convert the included `sample.vcf` to Hail's `.vds` format, run
```
$ ./build/install/hail/bin/hail importvcf src/test/resources/sample.vcf write -o ~/sample.vds
```
Then to count the number of samples and variants, run
```
$ ./build/install/hail/bin/hail read ~/sample.vds count
```
To compute and output sample and variant quality control statistics, run
```
$ ./build/install/hail/bin/hail read ~/sample.vds splitmulti variantqc -o ~/variantqc.tsv sampleqc -o ~/sampleqc.tsv
```
Note that during each run Hail writes a `hail.log` file in the current directory; this is useful to developers for debugging.

### Spark cluster

Hail is compatible with Spark versions 1.5 and 1.6, and uses 1.6.2 by default. Run
```
$ ./gradlew -Dspark.version=VERSION shadowJar
```
using the version of Spark on your cluster. The resulting JAR `build/libs/hail-all-spark.jar` can be submitted using `spark-submit`. See [Spark documentation](http://spark.apache.org/docs/1.6.2/cluster-overview.html) for details.

### BLAS and LAPACK

Hail uses BLAS and LAPACK optimized linear algebra libraries. On Linux, these must be explicitly installed. On Ubuntu 14.04, run
```
$ apt-get install libatlas-base-dev
```
If natives are not found, `hail.log` will contain the warnings
```
Failed to load implementation from: com.github.fommil.netlib.NativeSystemLAPACK
Failed to load implementation from: com.github.fommil.netlib.NativeSystemBLAS
```
See [netlib-java](http://github.com/fommil/netlib-java) for more information.

## Running the tests

Several Hail tests have additional dependencies:

- PLINK 1.9, [http://www.cog-genomics.org/plink2](http://www.cog-genomics.org/plink2)

- QCTOOL 1.4, [http://www.well.ox.ac.uk/~gav/qctool](http://www.well.ox.ac.uk/~gav/qctool)

- R 3.3.1, [http://www.r-project.org/](http://www.r-project.org/) with packages `jsonlite`, `SKAT`, and `logistf`, which depends on `mice` and `Rcpp`.

Other recent versions of QCTOOL and R should suffice, but PLINK 1.0 will not.

To execute all Hail tests, run
```
$ ./gradlew test
```