===============
Microsoft Azure
===============

``hailctl hdinsight``
---------------------

As of version 0.2.82, pip installations of Hail come bundled with a command-line tool, ``hailctl
hdinsight`` for working with `Microsoft Azure HDInsight Spark
<https://docs.microsoft.com/en-us/azure/hdinsight/hdinsight-overview>`__ clusters configured for
Hail.

This tool requires the `Azure CLI <https://docs.microsoft.com/en-us/cli/azure/install-azure-cli>`__.

An HDInsight cluster always consists of two "head" nodes, two or more "worker" nodes, and an Azure
Blob Storage container. The head nodes are automatically configured to serve Jupyter Notebooks at
``https://CLUSTER_NAME.azurehdinsight.net/jupyter`` . The Jupyter server is protected by a
username-password combination. The username and password are printed to the terminal after the
cluster is created.

Every HDInsight cluster is associated with one storage account which your Jupyter notebooks may
access. In addition, HDInsight will create a container within this storage account (sharing a name
with the cluster) for its own purposes. When a cluster is stopped using ``hailctl hdinsight stop``,
this container will be deleted.

To start a cluster, you must specify the cluster name, a storage account, and a resource group. The
storage account must be in the given resource group.

.. code-block:: text

    hailctl hdinsight start CLUSTER_NAME STORAGE_ACCOUNT RESOURCE_GROUP

To submit a Python job to that cluster, use:

.. code-block:: text

    hailctl hdinsight submit CLUSTER_NAME STORAGE_ACCOUNT HTTP_PASSWORD SCRIPT [optional args to your python script...]

To list running clusters:

.. code-block:: text

    hailctl hdinsight list

Importantly, to shut down a cluster when done with it, use:

.. code-block:: text

    hailctl hdinsight stop CLUSTER_NAME STORAGE_ACCOUNT RESOURCE_GROUP

.. _vep_hdinsight:

Variant Effect Predictor (VEP)
------------------------------

The following cluster configuration enables Hail to run VEP in parallel on every
variant in a dataset containing GRCh37 variants:

.. code-block:: text

    hailctl hdinsight start CLUSTER_NAME STORAGE_ACCOUNT RESOURCE_GROUP \
            --vep GRCh37 \
            --vep-loftee-uri https://STORAGE_ACCOUNT.blob.core.windows.net/CONTAINER/loftee-GRCh37 \
            --vep-homo-sapiens-uri https://STORAGE_ACCOUNT.blob.core.windows.net/CONTAINER/homo-sapiens-GRCh37

Those two URIs must point at directories containing the VEP data files. You can populate them by
downloading the two tar files using ``gcloud storage cp``,
``gs://hail-us-vep/loftee-beta/GRCh37.tar`` and ``gs://hail-us-vep/homo-sapiens/85_GRCh37.tar``,
extracting them into a local folder, and uploading that folder to your storage account using ``az
storage copy``. The hail-us-vep Google Cloud Storage bucket is a *requester pays* bucket which means
*you* must pay the cost of transferring them out of Google Cloud. We do not provide these files in
Azure because Azure Blob Storage lacks an equivalent cost control mechanism.

Hail also supports VEP for GRCh38 variants. The required tar files are located at
``gs://hail-REGION-vep/loftee-beta/GRCh38.tar`` and
``gs://hail-REGION-vep/homo-sapiens/95_GRCh38.tar``.

A cluster started without the ``--vep`` argument is unable to run VEP and cannot be modified to run
VEP. You must start a new cluster using ``--vep``.
