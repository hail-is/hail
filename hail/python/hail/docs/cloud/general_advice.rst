==============
General Advice
==============

Start Small
-----------

The cloud has a reputation for easily burning lots of money. You don't want to be the person who
spent ten thousand dollars one night without thinking about it. Luckily, it's easy to not be that person!

Always start small. For Hail, this means using a two worker Spark cluster and experimenting on a small 
fraction of the data. For genetic data, make sure your scripts work on chromosome 22 (the 2nd smallest autosomal chromosome) before
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
---------------

Estimating the time and cost of a Hail operation is often simple. Start a small cluster and use ``filter_rows`` to read a small fraction of the data:

.. code-block:: python

    test_mt = mt.filter_rows(mt.locus.contig == '22')
    print(mt.count_rows() / test_mt.count_rows())

Multiply the time spent computing results on this smaller dataset by the number printed. This yields a reasonable expectation of the time
to compute results on the full dataset using a cluster of the same size. However, not all operations will scale this way. Certain complicated operations
like ``pca`` or ``BlockMatrix`` multiplies do not scale linearly. When doing small time estimates, it can sometimes be helpful to get a few datapoints as
you gradually increase the size of your small dataset to see if it's scaling linearly.

Estimating cost
---------------

Costs vary between cloud providers. This cost estimate is based on Google Cloud, but the same principles often apply to other providers.

Google charges by the core-hour, so we can convert so-called "wall clock time" (time elapsed from starting the cluster to stopping the cluster)
to dollars-spent by multiplying it by the number of cores of each type and the price per core per hour of each type. At time of writing,
preemptible cores are 0.01 dollars per core hour and non-preemptible cores are 0.0475 dollars per core hour. Moreover, each core has an
additional 0.01 dollar "dataproc premium" fee. The cost of CPU cores for a cluster with an 8-core leader node; two non-preemptible, 8-core workers;
and 10 preemptible, 8-core workers running for 2 hours is:

.. code-block:: text

    2 * (2  * 8 * 0.0575 +  # non-preemptible workers
         10 * 8 * 0.02   +  # preemptible workers
         1  * 8 * 0.0575)   # leader (master) node

2.98 USD.

There are additional charges for persistent disk and SSDs. If your leader node has 100 GB and your worker nodes have 40 GB each you can expect
a modest increase in cost, slightly less than a dollar. The cost per disk is prorated from a per-month rate; at time of writing it is 0.04 USD
per GB per month. SSDs are more than four times as expensive.

In general, once you know the wall clock time of your job, you can enter your cluster parameters into the 
`Google Cloud Pricing Calculator <https://cloud.google.com/products/calculator/>`_. and get a precise estimate
of cost using the latest prices.
