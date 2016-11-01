# Hail

[![Gitter](https://badges.gitter.im/hail-is/hail.svg)](https://gitter.im/hail-is/hail?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge)

[Hail](https://hail.is) is an open-source scalable framework for exploring and analyzing genetic data. Starting from your [VCF](https://samtools.github.io/hts-specs/VCFv4.2.pdf) or similar format for sequencing or microarray data, Hail can:

 - generate variant quality control metrics like Hardy-Weinberg equilibrium or population-specific allele count
 - generate sample quality control metrics like imputed sex or the insertion/deletion ratio
 - compute principal components for each sample using your data, or project your cohort onto known ancestry coordinates from public datasets
 - import and manipulate variant and sample annotations imported from text tables, JSON, VCF, or locus interval files
 - produce new annotations computed from existing annotations and the genotype data
 - perform linear and logistic regressions to associate loci with phenotypes
 
The above examples (just a sample of Hail's flexibility!) are all backed by distributed algorithms that can analyze gigabyte-scale data on a laptop or terabyte-scale data on a large cloud cluster.

The Hail project began in late 2015, but already has been used for [research](http://biorxiv.org/content/early/2016/06/06/050195) and in large-scale genomics efforts including [ExAC v2](http://exac.broadinstitute.org/) and [gnomAD](http://gnomad.broadinstitute.org/). At the same time, Hail is under very active development as we work toward a stable release, so we do not guarantee forward compatibility of formats and interfaces. 

To get started using Hail:

- read the docs ([Getting Started](https://hail.is/getting_started.html), [Overview](https://hail.is/overview.html), [Tutorial](https://hail.is/tutorial.html), [General Reference](https://hail.is/reference.html), [Command Reference](https://hail.is/commands.html), [FAQ](https://hail.is/faq.html))
- join the [discussion forum](http://discuss.hail.is): we encourage this resource to be used for user/dev support 
- chat with the Hail team and other users in the [Hail Gitter](https://gitter.im/hail-is/hail) room

We encourage use of the [discussion forum](http://discuss.hail.is) for user and dev support, feature requests, and sharing your Hail-powered science. Please report any suspected bugs to [github issues](https://github.com/hail-is/hail/issues).

Want to get involved in development? Check out the [Github repo](https://github.com/hail-is/hail) and chat with us in the [Gitter dev room](https://gitter.im/hail-is/hail-dev).

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
