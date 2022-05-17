===================
Hail Query-on-Batch
===================

.. warning::

    Hail Query-on-Batch (the Batch backend) is currently in beta. This means some functionality is
    not yet working. Please `contact us <https://discuss.hail.is>`__ if you would like to use missing
    functionality on Query-on-Batch!


Hail Query-on-Batch uses Hail Batch instead of Apache Spark to execute jobs. Instead of a Dataproc
cluster, you will need a Hail Batch cluster. For more information on using Hail Batch, see the `Hail
Batch docs <https://hail.is/docs/batch/>`__. For more information on deploying a Hail Batch cluster,
please contact the Hail Team at our `discussion forum <https://discuss.hail.is>`__.

Getting Started
---------------

1. Install Hail version 0.2.93 or later:

.. code-block:: text

    pip install 'hail>=0.2.93'

2. `Sign up for a Hail Batch account <https://auth.hail.is/signup>`__ (currently only available to
   Broad affiliates).

3. Authenticate with Hail Batch.

.. code-block:: text

    hailctl auth login

3. Specify a bucket for Hail to use for temporary intermediate files. In Google Cloud, we recommend
   using a bucket with `automatic deletion after a set period of time
   <https://cloud.google.com/storage/docs/lifecycle>`__.

.. code-block:: text

    hailctl config set batch/tmp_dir gs://my-auto-delete-bucket/hail-query-temporaries

4. Specify a Hail Batch billing project (these are different from Google Cloud projects). Every new
   user has a trial billing project loaded with 10 USD. The name is available on the `Hail User
   account page <https://auth.hail.is/user>`__.

.. code-block:: text

    hailctl config set batch/billing_project my-billing-project

5. Set the default Hail Query backend to ``batch``:

.. code-block:: text

    hailctl config set query/backend batch

6. Now you are ready to `try Hail <../install/try.rst>`__! If you want to switch back to
   Query-on-Spark, run the previous command again with "spark" in place of "batch".

.. _vep_query_on_batch:

Variant Effect Predictor (VEP)
------------------------------

More information coming very soon. If you want to use VEP with Hail Query-on-Batch, please contact
the Hail Team at our `discussion forum <https://discuss.hail.is>`__.
