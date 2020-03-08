# Hail

Hail is an open-source, general-purpose, Python-based data analysis library with additional data types and methods for working with genomic data.

Hail is built to scale and has first-class support for multi-dimensional structured data, like the genomic data in a genome-wide association study (GWAS).

Hail's backend is implemented in Python, Scala, Java, and [Apache Spark](https://spark.apache.org/docs/latest/index.html).

See the [documentation](docs/0.2/) for more info on using
Hail. Post to the [Discussion Forum](http://discuss.hail.is) for user support
and feature requests. Chat with the Hail team and user community in Hail's
[Zulip chatroom](https://hail.zulipchat.com).

Hail is actively developed with new features and performance improvements integrated weekly. See the [changelog](docs/0.2/change_log.html) for more information.

### Community

Hail has been widely adopted in academia and industry, including as the analysis platform for the [genome aggregation database](https://gnomad.broadinstitute.org) and [UK Biobank rapid GWAS](https://www.nealelab.is/uk-biobank). Learn more about [Hail-powered science](references.html).

### Maintainer

Hail is maintained by a team in the [Neale lab](https://nealelab.is/) at the [Stanley Center for Psychiatric Research](http://www.broadinstitute.org/stanley) of the [Broad Institute of MIT and Harvard](http://www.broadinstitute.org) and the [Analytic and Translational Genetics Unit](https://www.atgu.mgh.harvard.edu/) of [Massachusetts General Hospital](http://www.massgeneral.org/).

Contact the Hail team at <a href="mailto:hail@broadinstitute.org"><code>hail@broadinstitute.org</code></a>.

### Citing Hail

If you use Hail for published work, please cite the software. You can get a citation for the version of Hail you installed by executing:

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
 - The Chan Zuckerburg Initiative

We would like to thank <a href="https://zulipchat.com/">Zulip</a> for supporting
open-source by providing free hosting, and YourKit, LLC for generously providing
free licenses for <a href="https://www.yourkit.com/java/profiler/">YourKit Java
Profiler</a> for open-source development.

<img src="https://www.yourkit.com/images/yklogo.png" align="right" />
