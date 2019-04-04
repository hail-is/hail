# Hail

Hail is an open-source, scalable framework for exploring and analyzing genomic data. 

For genomics applications, Hail can, for example:

 - flexibly [import and export](https://hail.is/docs/0.2/methods/impex.html) to a variety of data and annotation formats, including [VCF](https://samtools.github.io/hts-specs/VCFv4.2.pdf), [BGEN](http://www.well.ox.ac.uk/~gav/bgen_format/bgen_format_v1.2.html) and [PLINK](https://www.cog-genomics.org/plink2/formats)
 - generate variant annotations like call rate, Hardy-Weinberg equilibrium p-value, and population-specific allele count; and import annotations in parallel through [annotation datasets](https://hail.is/docs/stable/datasets.html), [VEP](https://useast.ensembl.org/info/docs/tools/vep/index.html), and [Nirvana](https://github.com/Illumina/Nirvana/wiki)
 - compute sample annotations like mean depth, imputed sex, and TiTv ratio
 - compute new annotations from existing ones as well as genotypes, and use these to filter samples, variants, and genotypes
 - find Mendelian violations in trios, prune variants in linkage disequilibrium, analyze genetic similarity between samples, and compute sample scores and variant loadings using PCA
 - perform variant, gene-burden and eQTL association analyses using linear, logistic, Poisson, and linear mixed regression, and estimate heritability
 - lots more! Check out some of the new features in [Hail 0.2](http://discuss.hail.is/t/announcing-hail-0-2/702/1).

Hail is a **[Python](https://www.python.org/)**  library with a scalable backend built on top of **[Apache Spark](https://spark.apache.org/docs/latest/index.html)** to efficiently analyze gigabyte-scale data on a laptop or terabyte-scale data on a cluster. 

### Getting Started

To get started using Hail:

- install Hail 0.2 using the instructions in [Installation](https://hail.is/docs/0.2/getting_started.html)
- follow the [Tutorials](https://hail.is/docs/0.2/tutorials-landing.html) for examples of how to use Hail
- read the [Hail Overview](https://hail.is/docs/0.2/overview.html) for a broad introduction to Hail
- check out the [Python API](https://hail.is/docs/0.2/api.html) for detailed information on the programming interface
- read over some of the [How-To materials](https://hail.is/docs/0.2/guides.html) for inspiration.

Hail uses a continuous deployment approach to software development, which means features, bug fixes, and performance improvements land every day. We recommend updating the software frequently.

### User Support

There are many ways to get in touch with the Hail team if you need help using Hail or would like to suggest improvements or new features.

- post to the [Discussion Forum](http://discuss.hail.is) for user support and feature requests.
- chat with the Hail team and user community in Hail's [Zulip chatroom](https://hail.zulipchat.com).
- please report any suspected bugs as [GitHub issues](https://github.com/hail-is/hail/issues)

### Maintainer

Hail is maintained by a team in the [Neale lab](https://nealelab.squarespace.com/) at the [Stanley Center for Psychiatric Research](http://www.broadinstitute.org/scientific-community/science/programs/psychiatric-disease/stanley-center-psychiatric-research/stanle) of the [Broad Institute of MIT and Harvard](http://www.broadinstitute.org) and the [Analytic and Translational Genetics Unit](https://www.atgu.mgh.harvard.edu/) of [Massachusetts General Hospital](http://www.massgeneral.org/).

Contact the Hail team: <a href="mailto:hail@broadinstitute.org"><code>hail@broadinstitute.org</code></a>.

Follow Hail on Twitter: <a href="https://twitter.com/hailgenetics">@hailgenetics</a>.

### Citing Hail

If you use Hail for published work, please cite the software:

 - Hail, https://github.com/hail-is/hail

### Acknowledgements

The Hail team has several sources of funding at the Broad Institute:

- The Stanley Center for Psychiatric Research, which together with Neale Lab has provided an incredibly supportive and stimulating home.
- Principal Investigators Benjamin Neale and Daniel MacArthur, whose scientific leadership has been essential for solving the right problems.
- Jeremy Wertheimer, whose strategic advice and generous philanthropy have been essential for growing the impact of Hail.

We are grateful for generous support from:

- The National Institute of Diabetes and Digestive and Kidney Diseases
- The National Institute of Mental Health
- The National Human Genome Research Institute
- The Chan Zuckerburg Initiative

We would like to thank <a href="https://zulipchat.com/">Zulip</a> for supporting
open-source by providing free hosting, and YourKit, LLC for generously providing
free licenses for <a href="https://www.yourkit.com/java/profiler/">YourKit Java
Profiler</a> for open-source development.

<img src="https://www.yourkit.com/images/yklogo.png" align="right" />
