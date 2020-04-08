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