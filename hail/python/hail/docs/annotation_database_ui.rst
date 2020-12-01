.. _Annotation Database:

===================
Annotation Database
===================

.. warning::
    All functionality described on this page is experimental and subject to
    change.

This database contains a curated collection of variant annotations in an
accessible and Hail-friendly format, for use in Hail analysis pipelines.

To incorporate these annotations in your own Hail analysis pipeline, select
which annotations you would like to query from the table below and then
copy-and-paste the Hail generated code into your own analysis script.

Check out the :class:`.DB` class documentation for more detail on creating an
annotation database instance and annotating a :class:`.MatrixTable` or a
:class:`.Table`.

.. rubric:: Google Cloud Storage

Note that these annotations are stored in :ref:`Requester Pays<GCP Requester
Pays>` buckets on Google Cloud Storage. Buckets are now available in both the
US and EU regions, so egress charges may apply if your cluster is outside of
the region specified when creating an annotation database instance.

To access these buckets on a cluster started with ``hailctl dataproc``, you
can use the additional argument ``--requester-pays-annotation-db`` as follows:

.. code-block:: text

    hailctl dataproc start my-cluster --requester-pays-allow-annotation-db

.. rubric:: Amazon S3

Annotation datasets are now shared via `Open Data on AWS <https://aws.amazon
.com/opendata/>`__ as well, and can be accessed by users running Hail on
AWS. Note that on AWS the annotation datasets are currently only available in
a bucket in the US region.

Database Query
--------------

Select annotations by clicking on the checkboxes in the table, and the
appropriate Hail command will be generated in the panel below.

In addition, a search bar is provided if looking for a specific annotation
within our curated collection.

Use the "Copy to Clipboard" button to copy the generated Hail code, and paste
the command into your own Hail script.

.. raw:: html
   :file: _static/annotationdb/annotationdb.html