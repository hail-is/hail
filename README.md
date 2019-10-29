# Hail

[![Zulip](https://img.shields.io/badge/zulip-join_chat-brightgreen.svg)](https://hail.zulipchat.com?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge)

[Hail](https://hail.is) is an open-source, general-purpose, Python-based data analysis tool with additional data types and methods for working with genomic data.

Hail is built to scale and has first-class support for multi-dimensional structured data, like the genomic data in a genome-wide association study (GWAS).

Hail is exposed as a Python library, using primitives for distributed queries and linear algebra implemented in Scala, [Spark](https://spark.apache.org/docs/latest/index.html), and increasingly C++.

See the [documentation](https://hail.is/docs/0.2/) for more info on using Hail.

### Community

Hail has been widely adopted in academia and industry, including as the analysis platform for the [genome aggregation database](https://gnomad.broadinstitute.org) and [UK Biobank rapid GWAS](https://www.nealelab.is/uk-biobank). Learn more about [Hail-powered science](hail/www/references.md).

### Contribute

If you'd like to discuss or contribute to the development of methods or infrastructure, please:

- see the [For Software Developers](https://hail.is/docs/0.2/getting_started_developing.html) section of the installation guide for info on compiling Hail
- chat with us about development in our [Zulip chatroom](https://hail.zulipchat.com)
- visit the [Development Forum](http://dev.hail.is) for longer-form discussions
<!--- - read [this post]() (coming soon!) for tips on submitting a successful Pull Request to our repository --->

Hail uses a continuous deployment approach to software development, which means we frequently add new features. We update users about changes to Hail via the [Discussion Forum](http://discuss.hail.is). We recommend creating an account on the Discussion Forum so that you can subscribe to these updates as well.

### Maintainer

Hail is maintained by a team in the [Neale lab](https://nealelab.is/) at the [Stanley Center for Psychiatric Research](http://www.broadinstitute.org/stanley) of the [Broad Institute of MIT and Harvard](http://www.broadinstitute.org) and the [Analytic and Translational Genetics Unit](https://www.atgu.mgh.harvard.edu/) of [Massachusetts General Hospital](http://www.massgeneral.org/).

Contact the Hail team at <a href="mailto:hail@broadinstitute.org"><code>hail@broadinstitute.org</code></a>.

### Citing Hail

If you use Hail for published work, please cite the software. You can get a
citation for the version of Hail you installed by executing:

```python
import hail as hl
print(hl.citation())
```

Which will look like:

```
Hail Team. Hail 0.2.13-81ab564db2b4. https://github.com/hail-is/hail/releases/tag/0.2.13.
```

##### Acknowledgements

The Hail team has several sources of funding at the Broad Institute:
- The Stanley Center for Psychiatric Research, which together with Neale Lab has provided an incredibly supportive and stimulating home.
- Principal Investigators Benjamin Neale and Daniel MacArthur, whose scientific leadership has been essential for solving the right problems.
- Jeremy Wertheimer, whose strategic advice and generous philanthropy have been essential for growing the impact of Hail.

We are grateful for generous support from:
- The National Institute of Diabetes and Digestive and Kidney Diseases
- The National Institute of Mental Health
- The National Human Genome Research Institute
- The Chan Zuckerberg Initiative

We would like to thank <a href="https://zulipchat.com/">Zulip</a> for supporting
open-source by providing free hosting, and YourKit, LLC for generously providing
free licenses for <a href="https://www.yourkit.com/java/profiler/">YourKit Java
Profiler</a> for open-source development.

<img src="https://www.yourkit.com/images/yklogo.png" align="right" />
