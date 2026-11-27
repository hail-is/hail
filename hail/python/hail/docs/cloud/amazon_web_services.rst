===================
Amazon Web Services
===================

``hailctl emr`` (experimental)
------------------------------

``hailctl emr`` starts Amazon EMR on EC2 clusters for Hail, much like ``hailctl dataproc`` does on
Google Cloud. It supports one EMR release, ``emr-spark-8.1.0`` (Spark 4.1 and Scala 2.13), and
Python 3.12.

Prerequisites
~~~~~~~~~~~~~

* The AWS CLI configured with a default region (``aws configure``).
* The EMR default roles (``aws emr create-default-roles``).
* A Hail wheel built for Spark 4.1, uploaded to S3. The cluster downloads it with its instance
  profile, so that profile must be able to read it, and your credentials must be able to write
  next to it (``hailctl`` stores its bootstrap script in a ``bootstrap/`` folder there).
* If the wheel's bucket (or any bucket your jobs use) is encrypted with a customer-managed KMS key
  (SSE-KMS), the cluster's instance profile (``EMR_EC2_DefaultRole`` by default) needs
  ``kms:Decrypt`` and ``kms:GenerateDataKey`` on that key. Buckets that use the default S3
  encryption (SSE-S3) need nothing extra.
* The cluster needs internet access (for example through a NAT gateway) to install Python
  packages.

Starting and using a cluster
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: text

    hailctl emr start my-cluster --wheel s3://my-bucket/hail-0.2.140-py3-none-any.whl

Clusters shut down after one idle hour; change this with ``--idle-timeout`` (seconds).
Use ``--subnet-id`` and ``--service-access-security-group`` to start a cluster in a private subnet.

Submit a Python script (an ``s3://`` URI, or a local file with ``--upload-to``) and wait for it:

.. code-block:: text

    hailctl emr submit CLUSTER_ID my_script.py --upload-to s3://my-bucket/scripts/

List or terminate clusters:

.. code-block:: text

    hailctl emr list
    hailctl emr stop CLUSTER_ID

Notes
~~~~~

Amazon EMR 8.1 reads and writes ``s3://`` paths with S3A, and ``hailctl emr`` enables the MagicV2
committer. After a failed job, remove the output path before retrying. Add an S3 lifecycle rule to
abort incomplete multipart uploads.

Variant Effect Predictor (VEP)
------------------------------

VEP on EMR is not supported yet.
