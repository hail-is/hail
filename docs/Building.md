# Building

To build Hail, you need the [Gradle](http://gradle.org/) build tool.

## Build locally

These are instructions to build Hail to run locally, not on a Spark
cluster.  This is primarly intended for development or working with
small datasets.  To build Hail to run on a Spark cluster, see
"Building a Spark application jar" below.

To build Hail locally, run:

```
~/hail $ gradle installDist
```
	
This will populate `build/install` with an installation of Hail.  Then
you can directly run `build/install/hail/bin/hail`.

### Running

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

## Build a Spark application jar

To build a Spark application Jar, run:

```
~/hail $ gradle shadowJar
```

This builds `build/libs/hail-all-spark.jar`.  This can be submitted to
a Spark cluster using `spark-submit`.
