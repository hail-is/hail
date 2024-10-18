===========================
Use Hail on Google Dataproc
===========================

Requirements
------------

Running Hail on Dataproc requires having both Hail and the Google Cloud CLI installed on your machine.

Installing Hail
------
First, install Hail on your `Mac OS X <macosx.rst>`__ or `Linux <linux.rst>`__ laptop or
desktop. The Hail pip package includes a tool called ``hailctl dataproc`` which starts, stops, and
manipulates Hail-enabled Dataproc clusters.

Installing and configuring the Google Cloud SDK
------------

We recommend that you follow the `Google Cloud SDK documentation <https://cloud.google.com/sdk/docs/install>`__ to
install the Google Cloud SDK.

You will need to configure your Google Cloud SDK after installation. This is the time to set up your Google Cloud project
and billing, if you don't already have one.

Running Hail on Dataproc requires passing in a Dataproc region.

If you'd like to set your Dataproc region globally, you can do so by running:

.. code-block:: sh

    gcloud config set dataproc/region <your-region>


Otherwise, you can set your Dataproc region using the `hailctl` `--region` command line flag.

Starting your first Dataproc cluster
--------------

Start a dataproc cluster named "my-first-cluster". Cluster names may only
contain a mix lowercase letters and dashes. Starting a cluster can take as long
as two minutes.

.. code-block:: sh

   hailctl dataproc start my-first-cluster


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

   hailctl dataproc submit my-first-cluster hail-script.py

When the script is done running you'll see 25 rows of variant association
results.

You can also start a Jupyter Notebook running on the cluster:

.. code-block:: sh

   hailctl dataproc connect my-first-cluster notebook

When you are finished with the cluster stop it:

.. code-block:: sh

   hailctl dataproc stop my-first-cluster

Next Steps
""""""""""

- Read more about Hail on `Google Cloud <../cloud/google_cloud.rst>`__
- Get the `Hail cheatsheets <../cheatsheets.rst>`__
- Follow the Hail `GWAS Tutorial <../tutorials/01-genome-wide-association-study.rst>`__
