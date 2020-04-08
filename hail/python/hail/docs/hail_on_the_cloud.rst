.. _sec-hail_on_the_cloud:

=================
Hail on the Cloud
=================

Public clouds are a natural place to run Hail, offering the ability to run
on-demand workloads with high elasticity. For example, Google and Amazon make it
possible to rent Spark clusters with many thousands of cores on-demand,
providing for the elastic compute requirements of scientific research without
an up-front capital investment in hardware.

General Advice
--------------

Start Small
~~~~~~~~~~~

The cloud has a reputation for easily burning lots of money. You don't want to be the person who
spent ten thousand dollars one night without thinking about it. Luckily, it's easy to not be that person!

Always start small. For Hail, this means using a two worker Spark cluster and experimenting on a small 
fraction of the data. For genetic data, make sure your scripts work on chromosome 22 (the smallest one) before
you try running on the entire genome! If you have a matrix table you can limit to chromosome 22 with ``filter_rows``.
Hail will make sure not to load data for other chromosomes.

.. code-block:: python

    import hail as hl

    mt = hl.read_matrix_table('gs://....')
    mt = mt.filter_rows(mt.locus.contig == '22')

Hail's ``hl.balding_nichols_model`` creates a random genotype dataset with configurable numbers of rows and columns. 
You can use these datasets for experimentation.

As you'll see later, the smallest Hail cluster (on GCP) costs about 3 dollars per hour. Each time you think you need to double
the size of your cluster ask yourself: am I prepared to spend twice as much money per hour?

Estimating time
~~~~~~~~~~~~~~~

Estimating the time and cost of a Hail operation is often simple. Start a small cluster and use ``filter_rows`` to read a small fraction of the data:

.. code-block:: python

    test_mt = mt.filter_rows(mt.locus.contig == '22')
    print(mt.count_rows() / test_mt.count_rows())

Multiply the time spent computing results on this smaller dataset by the number printed. This yields a reasonable expectation of the time
to compute results on the full dataset using a cluster of the same size. However, not all operations will scale this way. Certain complicated operations
like ``pc_relate`` or ``BlockMatrix`` multiplies do not scale linearly. When doing small time estimates, it can sometimes be helpful to get a few datapoints as
you gradually increase the size of your small dataset to see if it's scaling linearly.

Estimating cost
~~~~~~~~~~~~~~~

Costs vary between cloud providers. This cost estimate is based on Google Cloud, but the same principles often apply to other providers.

Google charges by the core-hour, so we can convert so-called "wall clock time" (time elapsed from starting the cluster to stopping the cluster)
to dollars-spent by multiplying it by the number of cores of each type and the price per core per hour of each type. At time of writing,
preemptible cores are 0.01 dollars per core hour and non-preemptible cores are 0.0475 dollars per core hour. Moreover, each core has an
additional 0.01 dollar "dataproc premium" fee. The cost of CPU cores for a cluster with an 8-core leader node; two non-preemptible, 8-core workers;
and 10 preemptible, 8-core workers running for 2 hours is:

.. code-block:: text

    2 * (2  * 8 * 0.0575 +  # non-preemptible workers
     10 * 8 * 0.02 +   # preemptible workers
     1  * 8 * 0.0575)   # master node

Google Cloud Platform
---------------------

If you're new to Google Cloud in general, and would like an overview, linked 
`here <https://github.com/danking/hail-cloud-docs/blob/master/how-to-cloud.md>`__.
is a document written to onboard new users within our lab to cloud computing.

``hailctl``
~~~~~~~~~~~

As of version 0.2.15, pip installations of Hail come bundled with a command-line
tool, ``hailctl``. This tool has a submodule called ``dataproc``, the successor
to `Liam Abbott's cloudtools <https://github.com/Nealelab/cloudtools>`__, for
working with `Google Dataproc <https://cloud.google.com/dataproc/>`__ clusters
configured for Hail.

This tool requires the `Google Cloud SDK <https://cloud.google.com/sdk/gcloud/>`__.

Until full documentation for the command-line interface is written, we encourage
you to run the following command to see the list of modules:

.. code-block:: text

    hailctl dataproc

It is possible to print help for a specific command using the ``help`` flag:

.. code-block:: text

    hailctl dataproc start --help

To start a cluster, use:

.. code-block:: text

    hailctl dataproc start CLUSTER_NAME [optional args...]

To submit a Python job to that cluster, use:

.. code-block:: text

    hailctl dataproc submit CLUSTER_NAME SCRIPT [optional args to your python script...]

To connect to a Jupyter notebook running on that cluster, use:

.. code-block:: text

    hailctl dataproc connect CLUSTER_NAME notebook [optional args...]

To list active clusters, use:

.. code-block:: text

    hailctl dataproc list

Importantly, to shut down a cluster when done with it, use:

.. code-block:: text

    hailctl dataproc stop CLUSTER_NAME

Amazon Web Services
-------------------

While Hail does not have any built-in tools for working with
`Amazon EMR <https://aws.amazon.com/emr/>`__, we recommend the `open-source
tool <https://github.com/hms-dbmi/hail-on-AWS-spot-instances>`__ developed by Carlos De Niz
with the `Avillach Lab <https://avillach-lab.hms.harvard.edu/>`_ at Harvard Medical School

Other Cloud Providers
---------------------

There are no known open-source resources for working with Hail on cloud
providers other than Google and AWS. If you know of one, please submit a pull
request to add it here!

If you have scripts for working with Hail on other cloud providers, we may be
interested in including those scripts in ``hailctl`` (see above) as new
modules. Stop by the `dev forum <https://dev.hail.is>`__ to chat!