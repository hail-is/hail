# Hail for New Engineers

Hail exists to accelerate research on the genetics of human disease. We originally focused on the
needs of statistical geneticists working with very large human genetic datasets. These datasets
motivated the "Hail Query" project, which we describe later.

## Genetics, Briefly

Hail team is a part of Ben Neale's lab which is part of the Analytical and Translational Genetics
Unit (ATGU) at the Massachusetts General Hospital (MGH) and part of the Stanley Center (SC) at the
Broad Institute (Broad). Ben's lab studies, among other things, [statistical
genetics](https://en.wikipedia.org/wiki/Statistical_genetics) (statgen). The Broad Institute's
self-description, which follows, highlights a different field of study: genomics.

> [The] Broad Institute of MIT and Harvard was launched in 2004 to improve human health by using
> genomics to advance our understanding of the biology and treatment of human disease, and to help
> lay the groundwork for a new generation of therapies.

Genetics and genomics may seem similar to a software engineer's ear; however, genetics is the study
of heredity whereas genomics is the study of the genome.[^1] The [history of
genetics](https://en.wikipedia.org/wiki/History_of_genetics) is deeply intertwined with statistics
which perhaps explains some of the distinction from genomics whose history lies more firmly in
biology.

The history of genetics is also deeply intertwined with
[eugenics](https://en.wikipedia.org/wiki/History_of_eugenics) and
[racism](https://en.wikipedia.org/wiki/Scientific_racism). Sadly, this continues today (see: [James
Watson](https://en.wikipedia.org/wiki/James_Watson)). The team Zulip channel and private messages
are both valid forums for discussing these issues.

The Neale Lab manifests the Broad's credo by studying the relationship between human disease and
human genetics. This is sometimes called "genetic epidemiology". One common
statistical-epidemiological study design is the case-control study. A case-control study involves
two groups of people. The "case" group has been diagnosed with the disease. The "control" group has
not. We collect genetic material from both groups and search for a correlation between the material
and the groups.

There is at least one successful example of genetic studies leading to the development of a drug:
the discovery of PCSK9. In 2013, the New York Times [reported on the
discovery](https://www.nytimes.com/2013/07/10/health/rare-mutation-prompts-race-for-cholesterol-drug.html)
of an association between mutations in the PCSK9 gene and high levels of LDL cholesterol. By 2017,
[further
studies](https://www.nytimes.com/2017/03/17/health/cholesterol-drugs-repatha-amgen-pcsk9-inhibitors.html)
demonstrated *remarkable* reduction in LDL cholesterol levels. Unfortunately, as late as 2020, these
drugs [remain extraordinarily
expensive](https://www.nytimes.com/2018/10/02/health/pcsk9-cholesterol-prices.html).

## A Brief History

Around 2015, human genetic datasets had grown so large that analysis on a single computer was
prohibitively time-consuming. Moreover, partitioning the dataset and analyzing each partition
separately necessitated increasingly complex software engineering. We started the Hail project to
re-enable simple and interactive (i.e. fast) analysis of these large datasets.

Hail was a command-line program that used Apache Spark to run analysis on partitioned genetic
datasets simultaneously using hundreds of computer cores. To use Hail, a user needs an Apache Spark
cluster. Most Hail users use Google Dataproc Spark clusters.

The essential feature of a human genetic dataset is a two-dimensional matrix of human
genotypes. Every genotype has the property "number of alternate alleles". This property allows a
matrix of genotypes to be represented as a numeric matrix. Geneticists use linear algebraic
techniques on this numeric matrix to understand the relationship between human disease and human
genetics.

In November of 2016, the Hail command-line interface was eliminated and a Python interface was
introduced. During this time, Hail was not versioned. Users had to build and use Hail directly from
the source code repository.

In March of 2017, Hail team began work on a compiler.

In October of 2018, the Hail Python interface was modified, in a backwards-incompatible way. This
new Python interface was named "Hail 0.2". The old Python interface was retroactively named "Hail
0.1". Hail team began deploying a Python package named `hail` to [PyPI](https://pypi.org). The Hail
python package version complies with [Semantic Versioning](https://semver.org).

Meanwhile, in September of 2018, Hail team began work on a system called "Batch". Batch runs
programs in parallel on a cluster of virtual machines. Also in September, Hail team began work on a
system called "CI" (Continuous Integration). CI runs the tests for every pull request (PR) into the
`main` branch of [`hail-is/hail`](https://github.com/hail-is/hail). CI automatically merges into
main any pull request that both passes the tests and has at least one "approving" review and no
"changes requested" reviews. CI uses Hail Batch to run the tests.

Around this time, the Hail team organized itself into two sub-teams: "compilers" team and "services"
team. The compilers team is responsible for the Hail Python library, the compiler, and the
associated runtime. The compilers team code is entirely contained in the `hail` directory of the
`hail-is/hail` repository. The services team is responsible for Batch, CI, and the software
infrastructure that supports those services. Each service has its own directory in the hail
repository. More information about the structure of the repository can be found in
[`hail-overview.md`](hail-overview.md).

Beginning in early 2020, beta users were given access to Hail Batch.

In April of 2020, the Hail team began referring to the Hail python library as "Hail Query". The
"Hail Query Service" refers to the Hail Query python library using Hail Batch to run an analysis
across many computer cores instead of using Apache Spark.

## Hail Products, Briefly

Hail team maintains two software systems which our users directly use: Query and Batch

### Hail Query

Hail Query is a Python library for the analysis of large datasets. In Hail Query, a dataset is
represented as Table or a Matrix Table.

Hail Tables are similar to SQL tables, Pandas Dataframes, Excel spreadsheets, and CSV files.

Hail Matrix Tables do not have analogues in most other systems. Perhaps the only analogue is
[SciDB](https://dbdb.io/db/scidb) and its descendants: [TileDB](https://tiledb.com) and
[GenomicsDB](https://github.com/GenomicsDB/GenomicsDB)). A Hail Matrix Table can represent dense,
two-dimensional, homogeneous data. For example, datasets of Human genetic sequences, dense [numeric
matrices](https://en.wikipedia.org/wiki/Matrix_(mathematics)), and [astronomical
surveys](https://en.wikipedia.org/wiki/Astronomical_survey).

Users use the Hail Query python library to write a "pipeline" to analyze their data. The Python
library sends this pipeline to a compiler written in Scala. The compiler converts the pipeline into
an Apache Spark job or a Hail Batch job. A pipeline typically reads a dataset, processes it, and
either writes a new dataset or aggregates (e.g. computes the mean of a field). If a pipeline
aggregates, the resulting aggregated value is converted to a Python value and returned to the user.

### Hail Batch

Hail Batch is a system for executing arbitrary Linux programs. Each invocation of a program is
called a "job". Zero or more jobs comprise a Batch. Moreover, jobs may depend on the files written
by other jobs in the same Batch. The job dependencies are allowed to form any [directed, acyclic
graph (DAG)](https://en.wikipedia.org/wiki/Directed_acyclic_graph).

Users create batches and jobs using a Python library: `hailtop.batch`.

The [Batch Service](https://batch.hail.is) schedules jobs on an
[elastically](https://en.wikipedia.org/wiki/Elasticity_(cloud_computing)) sized group of virtual
machines. The virtual machines are often called "batch workers". The software that manages a single
virtual machine is also called "the batch worker".

## Infrastructure and Technology

The Hail team does not maintain any physical computer infrastructure.

We maintain some virtual infrastructure, almost exclusively within the Google Cloud Platform (GCP). These include:
- a [Kubernetes](https://kubernetes.io) (k8s) cluster called `vdc` (Virtual Data Center)
- many Google Cloud Storage ([an object store](https://en.wikipedia.org/wiki/Object_storage)) buckets
- one Cloud SQL instance with a production database, ephemeral pull-request-test databases, and a
  database per developer
- logs for basically anything can be found in [GCP Logs](https://console.cloud.google.com/logs)

We use a number of technologies:
- Python is the language of choice for web applications, services, and anything user-facing
- Scala is the legacy language of the Hail Query compiler and run-time
- the JVM is the legacy target of the Hail Query compiler
- C++ is the aspirational language of high-performance services and the Hail Query compiler and run-time
- LLVM is the aspirational target of the Hail Query compiler
- Docker is our container image and run-time system
- MySQL is our SQL database of choice

### Services Technology

We almost exclusively write services in Python 3.8. We use a number of Python packages:
- [`asyncio`](https://docs.python.org/3.8/library/asyncio.html) for concurrency which is built on
  [coroutines](https://en.wikipedia.org/wiki/Coroutine) not threads
- [`aiohttp`](https://docs.aiohttp.org/en/stable/) for serving HTTPS requests (most services speak
  HTTPS)
- [`jinja2`](https://jinja.palletsprojects.com/en/2.11.x/) for "templating" which simply means
  programmatically generating text files

A service is realized as:

- a Docker image containing the executable code for the service
- a Kubernetes deployment (which defines the command to run, how much CPU is needed, what
  environment variables are set, etc.)
- a Kubernetes service (which defines which ports are accessible)
- possibly a database within our Cloud SQL instance

We call a complete, working set of all services and databases a "namespace". Every namespace
corresponds to a Kubernetes namespace. All namespaces share one CloudSQL instance, but only have
access to their databases.

The default namespace contains our "production" services and is accessible to the outside world at
https://hail.is, https://batch.hail.is, etc.

Every developer and every pull request test job also has a namespace. Developer namespaces are
accessible at https://internal.hail.is/DEVELOPER_USERNAME/SERVICE/ . Unlike the default namespace,
every other namespace has exactly one database (containing all tables from each service's database).

All incoming traffic passes through either gateway or internal-gateway which route requests to
the appropriate namespace and service. Traffic from the Internet enters the cluster through gateway,
while traffic from Batch workers enters through internal-gateway.


[^1]: https://www.who.int/genomics/geneticsVSgenomics/en/
