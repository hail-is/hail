.. _Annotation Database:

===================
Annotation Database
===================

This database contains a curated collection of variant annotations in an
accessible and Hail-friendly format, for use in Hail analysis pipelines.

To incorporate these annotations in your own Hail analysis pipeline, select
which annotations you would like to query from the table below and then
copy-and-paste the Hail generated code into your own analysis script.

Note that most of these annotations are stored in :ref:`Requester Pays<GCP
Requester Pays>` buckets. These buckets are all stored in the US, so egress
charges may apply if your cluster is outside of the US.

To access these buckets on a cluster started with ``hailctl dataproc``, you
can use the additional argument ``--requester-pays-annotation-db`` as follows:

.. code-block:: text

    hailctl dataproc start my-cluster --requester-pays-allow-annotation-db

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