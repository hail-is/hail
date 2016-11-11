

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

### Running locally
The single command
```
$ ./gradlew installDist
```
installs Hail at `build/install/hail`. The initial build takes time as [Gradle](https://gradle.org/) installs all Hail dependencies. The executable is `build/install/hail/bin/hail` (to run using `hail` add `build/install/hail/bin` to your path).

Here are a few simple things to try in order. To list all commands, run
```
$ ./build/install/hail/bin/hail -h
```
To [import](https://hail.is/reference.html#Importing) the included `sample.vcf` into Hail's `.vds` format, run
```
$ ./build/install/hail/bin/hail \
    importvcf src/test/resources/sample.vcf \
    write -o ~/sample.vds
```
To [split](https://hail.is/commands.html#splitmulti) multi-allelic variants, compute a panel of [sample](https://hail.is/commands.html#sampleqc) and [variant](https://hail.is/commands.html#variantqc) quality control statistics, write these statistics to files, and save an annotated version of the vds, run:
```
$ ./build/install/hail/bin/hail \
    read ~/sample.vds \
    splitmulti \
    sampleqc -o ~/sampleqc.tsv \
    variantqc \
    exportvariants -o ~/variantqc.tsv -c 'Variant = v, va.qc.*' \
    write -o ~/sample.qc.vds
```
To count the number of samples, variants, and genotypes, run:
```
$ ./build/install/hail/bin/hail read ~/sample.qc.vds \
    count --genotypes
```
Now let's get a feel for Hail's powerful [object methods](https://hail.is/reference.html#HailObjectProperties), [annotation system](https://hail.is/reference.html#Annotations), and [expression language](https://hail.is/reference.html#HailExpressionLanguage). To [print](https://hail.is/commands.html#printschema) the current annotation schema and use these annotations to [filter](https://hail.is/reference.html#Filtering) variants, samples, and genotypes, run:
```
$ ./build/install/hail/bin/hail read ~/sample.qc.vds \
    printschema -o ~/schema.txt \
    filtervariants expr --keep \
      -c 'v.altAllele.isSNP && va.qc.gqMean >= 20' \
    filtersamples expr --keep \
      -c 'sa.qc.callRate >= 0.97 && sa.qc.dpMean >= 15' \
    filtergenotypes --keep \
      -c 'let ab = g.ad[1] / g.ad.sum in
           ((g.isHomRef && ab <= 0.1) || 
            (g.isHet && ab >= 0.25 && ab <= 0.75) || 
            (g.isHomVar && ab >= 0.9))' \
    write -o ~/sample.filtered.vds
```
Try running `count` on `sample.filtered.vds` to see how the numbers have changed. For further background and examples, continue to the [overview](https://hail.is/overview.html), check out the [general reference](https://hail.is/reference.html) and [command reference](https://hail.is/commands.html), and try the [tutorial](https://hail.is/tutorial.html).

Note that during each run Hail writes a `hail.log` file in the current directory; this is useful to developers for debugging.

### Running on a Spark cluster

Hail is compatible with Spark versions 1.5 and 1.6, and uses 1.6.2 by default. Run
```
$ ./gradlew -Dspark.version=VERSION shadowJar
```
using the version of Spark on your cluster. The resulting JAR `build/libs/hail-all-spark.jar` can be submitted using `spark-submit`. See [Spark documentation](http://spark.apache.org/docs/1.6.2/cluster-overview.html) for details.

[Google](https://cloud.google.com/dataproc/) and [Amazon](https://aws.amazon.com/emr/details/spark/) offer optimized Spark performance and exceptional scalability to tens of thousands of cores without the overhead of installing and managing an on-prem cluster.
To get started running Hail on the Google Cloud Platform, see this [forum post](http://discuss.hail.is/t/using-hail-on-the-google-cloud-platform/80).

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
