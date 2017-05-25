# Hail

[![Gitter](https://badges.gitter.im/hail-is/hail.svg)](https://gitter.im/hail-is/hail?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge) [![CI Status](https://ci.hail.is/app/rest/builds/buildType:(id:HailSourceCode_HailCi)/statusIcon)](https://ci.hail.is/viewType.html?buildTypeId=HailSourceCode_HailCi&branch_HailSourceCode_HailMainline=%3Cdefault%3E&tab=buildTypeStatusDiv)

[Hail](https://hail.is) is an open-source, scalable framework for exploring and analyzing genomic data. Starting from sequencing or microarray data in [VCF](https://samtools.github.io/hts-specs/VCFv4.2.pdf) and [other formats](https://hail.is/hail/hail.HailContext.html#hail.HailContext), Hail can, for example:

 - generate variant annotations like call rate, Hardy-Weinberg equilibrium p-value, and population-specific allele count
 - generate sample annotations like mean depth, imputed sex, and TiTv ratio
 - load variant and sample annotations from text tables, JSON, VCF, VEP, and locus interval files
 - generate new annotations from existing annotations and the genotypes, and use these to filter samples, variants, and genotypes
 - find Mendelian violations in trios, prune variants in linkage disequilibrium, analyze genetic similarity between samples via the GRM and IBD matrix, and compute sample scores and variant loadings using PCA
 - perform variant, gene-burden and eQTL association analyses using linear, logistic, and linear mixed regression, and estimate heritability

All this functionality is exposed through **[Python](https://www.python.org/)** and backed by distributed algorithms built on top of **[Apache Spark](http://spark.apache.org/)** to efficiently analyze gigabyte-scale data on a laptop or terabyte-scale data on an on-prem cluster or in the cloud.

Hail is used in [published research](http://biorxiv.org/content/early/2016/06/06/050195) and as the core analysis platform of large-scale genomics efforts such as [gnomAD](http://gnomad.broadinstitute.org/). The project began in Fall 2015 to [harness the flood of genomic data](https://www.broadinstitute.org/blog/harnessing-flood-scaling-data-science-big-genomics-era) and is under very active development as we work toward a stable release, so we do not guarantee forward compatibility of formats and interfaces.

Want to get involved in development? Check out the [Github repo](https://github.com/hail-is/hail), chat with us in the [Gitter dev room](https://gitter.im/hail-is/hail-dev), view our keynote at [Spark Summit East 2017](https://spark-summit.org/east-2017/events/scaling-genetic-data-analysis-with-apache-spark/), or connect with us June 6-7 at [Spark Summit West 2017](https://spark-summit.org/2017/events/scaling-genetic-data-analysis-with-apache-spark/).

Or **come join us full-time** at the [Broad Institute of MIT and Harvard](https://www.broadinstitute.org/)! We are founding a new Initiative in Scalable Analytics and recruiting **software engineers** at multiple levels of experience. Details [here](https://www.linkedin.com/jobs/view/316818823/).

#### Getting Started

To get started using Hail on your data or [public data](https://console.cloud.google.com/storage/browser/genomics-public-data/):

- follow the installation instructions in [Getting Started](https://hail.is/hail/getting_started.html)
- check out the [Overview](https://hail.is/hail/overview.html), [Tutorial](https://hail.is/hail/tutorial.html), and [Python API](https://hail.is/hail/index.html)
- chat with the Hail team in the [Hail Gitter](https://gitter.im/hail-is/hail) room

We encourage use of the [Discussion Forum](http://discuss.hail.is) for user and dev support, feature requests, and sharing your Hail-powered science. Follow Hail on Twitter [@hailgenetics](https://twitter.com/hailgenetics). Please report any suspected bugs to [github issues](https://github.com/hail-is/hail/issues).

#### Hail Team

The Hail team is based in the [Neale lab](https://nealelab.squarespace.com/) at the [Stanley Center for Psychiatric Research](http://www.broadinstitute.org/scientific-community/science/programs/psychiatric-disease/stanley-center-psychiatric-research/stanle) of the [Broad Institute of MIT and Harvard](http://www.broadinstitute.org) and the [Analytic and Translational Genetics Unit](https://www.atgu.mgh.harvard.edu/) of [Massachusetts General Hospital](http://www.massgeneral.org/).

Contact the Hail team at
<a href="mailto:hail@broadinstitute.org"><code>hail@broadinstitute.org</code></a>.

#### Citing Hail

If you use Hail for published work, please cite the software:

 - Hail, [https://github.com/hail-is/hail](https://github.com/hail-is/hail)

and either the forthcoming manuscript describing Hail (if possible):

 - Cotton Seed, Alex Bloemendal, Jonathan M Bloom, Jacqueline I Goldstein, Daniel King, Timothy Poterba, Benjamin M. Neale.  _Hail: An Open-Source Framework for Scalable Genetic Data Analysis_.  In preparation.

or the following paper which includes a brief introduction to Hail in the online methods:

 - Andrea Ganna, Giulio Genovese, et al. _Ultra-rare disruptive and damaging mutations influence educational attainment in the general population_.  [Nature Neuroscience](http://www.nature.com/neuro/journal/vaop/ncurrent/full/nn.4404.html)

And we'd love to hear about your work in the [Science category](http://discuss.hail.is/c/science) of the discussion forum!
