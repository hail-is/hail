.. _sec-hail_on_the_cloud:

=================
Hail on the Cloud
=================

Public clouds are a natural place to run Hail, offering the ability to run
on-demand workloads with high elasticity.

Google Cloud Platform
---------------------

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

    hailctl dataproc submit CLUSTER_NAME SCRIPT [optional args...]

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