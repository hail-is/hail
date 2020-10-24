.. _sec-datasets:

========
Datasets
========

.. warning::
    All functionality described on this page is experimental and subject to
    change.

This page describes genetic datasets that are hosted in public buckets
on both Google Cloud Storage and AWS S3. Note that these datasets are stored in
:ref:`Requester Pays<GCP Requester Pays>` buckets on GCS, and are available in
both the US and EU regions.

Check out the :func:`.load_dataset` function to see how to load one of these
datasets into a Hail pipeline. For example, to load the
1000_Genomes_autosomes dataset from the US bucket on Google Cloud Storage:

.. code-block:: python

    >>> mt_1kg = hl.experimental.load_dataset(name='1000_Genomes_autosomes',
                                              version='phase_3',
                                              reference_genome='GRCh38',
                                              region='us',
                                              cloud='gcp')


View schemas for available datasets:

.. toctree::
    :maxdepth: 1

    datasets/schemas.rst

.. raw:: html
   :file: _static/datasets/datasets.html