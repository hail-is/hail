===========================
Use Hail on Azure HDInsight
===========================

First, install Hail on your `Mac OS X <macosx.rst>`__ or `Linux <linux.rst>`__ laptop or
desktop. The Hail pip package includes a tool called ``hailctl hdinsight`` which starts, stops, and
manipulates Hail-enabled HDInsight clusters.

Start an HDInsight cluster named "my-first-cluster". Cluster names may only contain lowercase
letters, uppercase letter, and numbers. You must already have a storage account and resource
group.

.. code-block:: sh

   hailctl hdinsight start MyFirstCluster MyStorageAccount MyResourceGroup


Be sure to record the generated http password so that you can access the cluster.

Create a file called "hail-script.py" and place the following analysis of a
randomly generated dataset with five-hundred samples and half-a-million
variants.

.. code-block:: python3

    import hail as hl
    mt = hl.balding_nichols_model(n_populations=3,
                                  n_samples=500,
                                  n_variants=500_000,
                                  n_partitions=32)
    mt = mt.annotate_cols(drinks_coffee = hl.rand_bool(0.33))
    gwas = hl.linear_regression_rows(y=mt.drinks_coffee,
                                     x=mt.GT.n_alt_alleles(),
                                     covariates=[1.0])
    gwas.order_by(gwas.p_value).show(25)

Submit the analysis to the cluster and wait for the results. You should not have
to wait more than a minute.

.. code-block:: sh

   hailctl hdinsight submit MyFirstCluster MyStorageAccount HTTP_PASSWORD MyResourceGroup hail-script.py

When the script is done running you'll see 25 rows of variant association
results.

You can also connect to a Jupyter Notebook running on the cluster at
https://MyFirstCluster.azurehdinisght.net/jupyter

When you are finished with the cluster stop it:

.. code-block:: sh

   hailctl hdinsight stop MyFirstCluster MyStorageAccount MyResourceGroup

Next Steps
""""""""""

- Read more about Hail on `Azure HDInsight <../cloud/azure.rst>`__
- Get the `Hail cheatsheets <../cheatsheets.rst>`__
- Follow the Hail `GWAS Tutorial <../tutorials/01-genome-wide-association-study.rst>`__
