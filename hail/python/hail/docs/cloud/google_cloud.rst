=====================
Google Cloud Platform
=====================

If you're new to Google Cloud in general, and would like an overview, linked 
`here <https://github.com/danking/hail-cloud-docs/blob/master/how-to-cloud.md>`__.
is a document written to onboard new users within our lab to cloud computing.

``hailctl dataproc``
--------------------

As of version 0.2.15, pip installations of Hail come bundled with a command-line
tool, ``hailctl``. This tool has a submodule called ``dataproc`` for working with
`Google Dataproc <https://cloud.google.com/dataproc/>`__ clusters configured for Hail.

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

Reading from Google Cloud Storage
---------------------------------

A dataproc cluster created through ``hailctl dataproc`` will automatically be configured to allow hail to read files from 
Google Cloud Storage (GCS). To allow hail to read from GCS when running locally, you need to install the 
`Cloud Storage Connector <https://cloud.google.com/dataproc/docs/concepts/connectors/cloud-storage>`_. The easiest way to do that is to
run the following script from your command line:

.. code-block:: text

    curl -sSL https://broad.io/install-gcs-connector | python3

After this is installed, you'll be able to read from paths beginning with ``gs`` directly from you laptop.

.. _GCP Requester Pays:

Requester Pays
--------------

Some google cloud buckets are `Requester Pays <https://cloud.google.com/storage/docs/requester-pays>`_, meaning 
that accessing them will incur charges on the requester. Google breaks down the charges in the linked document,
but the most important class of charges to be aware of are `Network Charges <https://cloud.google.com/storage/pricing#network-pricing>`_.
Specifically, the egress charges. You should always be careful reading data from a bucket in a different region
then your own project, as it is easy to rack up a large bill. For this reason, you must specifically enable 
requester pays on your ``hailctl dataproc`` cluster if you'd like to use it.

To allow your cluster to read from any requester pays bucket, use:

.. code-block:: text

    hailctl dataproc start CLUSTER_NAME --requester-pays-allow-all

To make it easier to avoid accidentally reading from a requester pays bucket, we also have
``--requester-pays-allow-buckets``. If you'd like to enable only reading from buckets named
``hail-bucket`` and ``big-data``, you can specify the following:

.. code-block:: text

    hailctl dataproc start  my-cluster --requester-pays-allow-buckets hail-bucket,big-data

Users of the :ref:`Annotation Database` will find that many of the files are stored in requester pays buckets.
In order to allow the dataproc cluster to read from them, you can either use ``--requester-pays-allow-all`` from above
or use the special ``--requester-pays-allow-annotation-db`` to enable the specific list of buckets that the annotation database
relies on.

.. _vep_dataproc:

Variant Effect Predictor (VEP)
------------------------------

The following cluster configuration enables Hail to run VEP in parallel on every
variant in a dataset containing GRCh37 variants:

.. code-block:: text

    hailctl dataproc start NAME --vep GRCh37

Hail also supports VEP for GRCh38 variants, but you must start a cluster with
the argument ``--vep GRCh38``. A cluster started without the ``--vep`` argument is
unable to run VEP and cannot be modified to run VEP. You must start a new
cluster using ``--vep``.
