# Hail

[![Zulip](https://img.shields.io/badge/zulip-join_chat-brightgreen.svg)](https://hail.zulipchat.com?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge) [![CI Status](https://ci.hail.is/app/rest/builds/buildType:(id:HailSourceCode_HailCi)/statusIcon)](https://ci.hail.is/viewType.html?buildTypeId=HailSourceCode_HailCi&branch_HailSourceCode_HailMainline=%3Cdefault%3E&tab=buildTypeStatusDiv)

[Hail](https://hail.is) is an open-source, scalable framework for exploring and analyzing genomic data. 

The Hail project began in Fall 2015 to empower the worldwide genetics community to [harness the flood of genomes](https://www.broadinstitute.org/blog/harnessing-flood-scaling-data-science-big-genomics-era) to discover the biology of human disease. Since then, Hail has expanded to enable analysis of large-scale datasets beyond the field of genomics. 

Here are two examples of projects powered by Hail:

- The [gnomAD](http://gnomad.broadinstitute.org/) team uses Hail as its core analysis platform. gnomAD is among the most comprehensive catalogues of human genetic variation in the world, and one of the largest genetic datasets. Analysis results are shared publicly and have had sweeping impact on biomedical research and the clinical diagnosis of genetic disorders.
- The Neale Lab at the Broad Institute used Hail to perform QC and genome-wide association analysis of 2419 phenotypes across 10 million variants and 337,000 samples from the UK Biobank in 24 hours. These results are also publicly available, see the Neale Lab [blog](http://www.nealelab.is/blog/) for more info.

For genomics applications, Hail can:

 - flexibly [import and export](https://hail.is/docs/devel/methods/impex.html) to a variety of data and annotation formats, including [VCF](https://samtools.github.io/hts-specs/VCFv4.2.pdf), [BGEN](http://www.well.ox.ac.uk/~gav/bgen_format/bgen_format_v1.2.html) and [PLINK](https://www.cog-genomics.org/plink2/formats)
 - generate variant annotations like call rate, Hardy-Weinberg equilibrium p-value, and population-specific allele count; and import annotations in parallel through the [annotation database](https://hail.is/docs/stable/annotationdb.html), [VEP](https://useast.ensembl.org/info/docs/tools/vep/index.html), and [Nirvana](https://github.com/Illumina/Nirvana/wiki)
 - generate sample annotations like mean depth, imputed sex, and TiTv ratio
 - generate new annotations from existing ones as well as genotypes, and use these to filter samples, variants, and genotypes
 - find Mendelian violations in trios, prune variants in linkage disequilibrium, analyze genetic similarity between samples, and compute sample scores and variant loadings using PCA
 - perform variant, gene-burden and eQTL association analyses using linear, logistic, and linear mixed regression, and estimate heritability
 - lots more!

Hail's functionality is exposed through **[Python](https://www.python.org/)** and backed by distributed algorithms built on top of **[Apache Spark](https://spark.apache.org/docs/latest/index.html)** to efficiently analyze gigabyte-scale data on a laptop or terabyte-scale data on a cluster. 

Users can script pipelines or explore data interactively in [Jupyter notebooks](http://jupyter.org/) that combine Hail's methods, PySpark's scalable [SQL](https://spark.apache.org/docs/latest/sql-programming-guide.html) and [machine learning algorithms](https://spark.apache.org/docs/latest/ml-guide.html), and Python libraries like [pandas](http://pandas.pydata.org/)'s [scikit-learn](http://scikit-learn.org/stable/) and [Matplotlib](https://matplotlib.org/). Hail also provides a flexible domain language to express complex quality control and analysis pipelines with concise, readable code.

To learn more, you can view our talks at [Spark Summit East](https://spark-summit.org/east-2017/events/scaling-genetic-data-analysis-with-apache-spark/) and [Spark Summit West](https://spark-summit.org/2017/events/scaling-genetic-data-analysis-with-apache-spark/) (below).

[![Hail talk at Spark Summit West 2017](https://storage.googleapis.com/hail-common/hail_spark_summit_west.png)](https://www.youtube.com/watch?v=pyeQusIN5Ao&list=PLlMMtlgw6qNjROoMNTBQjAcdx53kV50cS)

### Getting Started

There are currently two versions of Hail: `0.1` (stable) and `0.2 beta` (development).
We recommend that new users install `0.2 beta`, since this version is already radically improved from `0.1`,
 the file format is stable, and the interface is nearly stable.

To get started using Hail `0.2 beta` on your own data or on [public data](https://console.cloud.google.com/storage/browser/genomics-public-data/):

- install Hail using the instructions in [Getting Started](https://hail.is/docs/devel/getting_started.html)
- read the [Overview](https://hail.is/docs/devel/tutorials-landing.html) for a broad introduction to Hail
- follow the [Tutorials](https://hail.is/docs/devel/overview.html) for examples of how to use Hail
- check out the [Python API](https://hail.is/docs/devel/api.html) for detailed information on the programming interface

As we work toward a stable `0.2` release, additional improvements to the interface may require users to modify their pipelines
when updating to the latest patch. All such breaking changes will be logged [here](http://discuss.hail.is/t/log-of-breaking-changes-in-0-2-beta/454).

See the [Hail 0.1 docs](https://hail.is/docs/stable/index.html) to get started with `0.1`. The [Annotation Database](https://hail.is/docs/stable/annotationdb.html) and [gnomAD distribution](http://gnomad-beta.broadinstitute.org/downloads) are currently only directly available
for `0.1` but will be updated for `0.2` soon.

### User Support

There are many ways to get in touch with the Hail team if you need help using Hail, or if you would like to suggest improvements or features. We also love to hear from new users about how they are using Hail.

- chat with the Hail team in our [Zulip chatroom](https://hail.zulipchat.com).
- post to the [Discussion Forum](http://discuss.hail.is) for user support and feature requests, or to share your Hail-powered science 
- please report any suspected bugs to [github issues](https://github.com/hail-is/hail/issues)

Hail uses a continuous deployment approach to software development, which means we frequently add new features. We update users about changes to Hail via the Discussion Forum. We recommend creating an account on the Discussion Forum so that you can subscribe to these updates.

### Contribute

Hail is committed to open-source development. Our [Github repo](https://github.com/hail-is/hail) is publicly visible. If you'd like to contribute to the development of methods or infrastructure, please: 

- see [Getting Started Developing](https://hail.is/docs/devel/getting_started_developing.html) for info on compiling Hail
- chat with us about development in our [Zulip chatroom](https://hail.zulipchat.com)
- visit the [Development Forum](http://dev.hail.is) for longer-form discussions
<!--- - read [this post]() (coming soon!) for tips on submitting a successful Pull Request to our repository --->


### Hail Team

The Hail team is embedded in the [Neale lab](https://nealelab.squarespace.com/) at the [Stanley Center for Psychiatric Research](http://www.broadinstitute.org/scientific-community/science/programs/psychiatric-disease/stanley-center-psychiatric-research/stanle) of the [Broad Institute of MIT and Harvard](http://www.broadinstitute.org) and the [Analytic and Translational Genetics Unit](https://www.atgu.mgh.harvard.edu/) of [Massachusetts General Hospital](http://www.massgeneral.org/).

Contact the Hail team at <a href="mailto:hail@broadinstitute.org"><code>hail@broadinstitute.org</code></a>.

Follow Hail on Twitter <a href="https://twitter.com/hailgenetics">@hailgenetics</a>.

### Citing Hail

If you use Hail for published work, please cite the software:

 - Hail, [https://github.com/hail-is/hail](https://github.com/hail-is/hail)

##### Acknowledgements

We would like to thank <a href="https://zulipchat.com/">Zulip</a> for supporting
open-source by providing free hosting, and YourKit, LLC for generously providing
free licenses for <a href="https://www.yourkit.com/java/profiler/">YourKit Java
Profiler</a> for open-source development.

<img src="https://www.yourkit.com/images/yklogo.png" align="right" />