# Hail

[![Zulip](https://img.shields.io/badge/zulip-join_chat-brightgreen.svg)](https://hail.zulipchat.com?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge)

[Hail](https://hail.is) is an open-source, general-purpose, Python-based data analysis tool with additional data types and methods for working with genomic data. Hail is used throughout academia and industry as the analytical engine for major studies, projects, and services, including the Genome Aggregation Database ([gnomad.broadinstitute.org](http://gnomad.broadinstitute.org)) and Neale lab mega-GWAS ([nealelab.is/uk-biobank](https://nealelab.is/uk-biobank)).

Unlike the Python and R scientific computing stacks, Hail:

- scales from laptop to large compute cluster or cloud, with the same code
- is designed to work with datasets that do not fit in memory
- has first-class support for multi-dimensional structured data, like genomic data as in this [tutorial](https://hail.is/docs/0.2/tutorials/01-genome-wide-association-study.html)

Hail's methods are primarily written in Python, using primitives for distributed queries and linear algebra implemented in Scala, [Spark](https://spark.apache.org/docs/latest/index.html), and increasingly C++. We welcome the scientific community to leverage Hail to develop, share, and apply new methods at scale!

See the [homepage](https://hail.is) for more info on using Hail.

### Contribute

Hail is committed to open-source development. If you'd like to contribute to the development of methods or infrastructure, please: 

- see the [For Software Developers](https://hail.is/docs/0.2/getting_started_developing.html) section of the installation guide for info on compiling Hail
- chat with us about development in our [Zulip chatroom](https://hail.zulipchat.com)
- visit the [Development Forum](http://dev.hail.is) for longer-form discussions
<!--- - read [this post]() (coming soon!) for tips on submitting a successful Pull Request to our repository --->

Hail uses a continuous deployment approach to software development, which means we frequently add new features. We update users about changes to Hail via the [Discussion Forum](http://discuss.hail.is). We recommend creating an account on the Discussion Forum so that you can subscribe to these updates as well.

### Hail Team

The Hail team is embedded in the [Neale lab](https://nealelab.squarespace.com/) at the [Stanley Center for Psychiatric Research](http://www.broadinstitute.org/scientific-community/science/programs/psychiatric-disease/stanley-center-psychiatric-research/stanle) of the [Broad Institute of MIT and Harvard](http://www.broadinstitute.org) and the [Analytic and Translational Genetics Unit](https://www.atgu.mgh.harvard.edu/) of [Massachusetts General Hospital](http://www.massgeneral.org/).

Contact the Hail team at <a href="mailto:hail@broadinstitute.org"><code>hail@broadinstitute.org</code></a>.

Follow Hail on Twitter <a href="https://twitter.com/hailgenetics">@hailgenetics</a>.

### Citing Hail

If you use Hail for published work, please cite the software:

 - Hail, https://github.com/hail-is/hail

##### Acknowledgements

We would like to thank <a href="https://zulipchat.com/">Zulip</a> for supporting
open-source by providing free hosting, and YourKit, LLC for generously providing
free licenses for <a href="https://www.yourkit.com/java/profiler/">YourKit Java
Profiler</a> for open-source development.

<img src="https://www.yourkit.com/images/yklogo.png" align="right" />
