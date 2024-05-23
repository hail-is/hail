.. _sec-datasets:

========
Datasets
========

.. warning::
    All functionality described on this page is experimental and subject to
    change.

This page describes genetic datasets that are hosted in public buckets on both
Google Cloud Storage and Amazon S3. Note that these datasets are stored in
:ref:`Requester Pays<GCP Requester Pays>` buckets on GCS, and are available in
both the US-CENTRAL1 and EUROPE-WEST1 regions. On AWS, the datasets are shared
via `Open Data on AWS <https://aws.amazon.com/opendata/>`__ and are in buckets
in the US region.

Check out the :func:`.load_dataset` function to see how to load one of these
datasets into a Hail pipeline. You will need to provide the name, version, and
reference genome build of the desired dataset, as well as specify the region
your cluster is in and the cloud platform. Egress charges may apply if your
cluster is outside of the region specified.

.. rubric:: Schemas for Available Datasets

.. toctree::
    :maxdepth: 1

    datasets/schemas.rst

.. raw:: html
   :file: _static/datasets/datasets.html
