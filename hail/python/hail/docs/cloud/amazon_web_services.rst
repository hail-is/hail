===================
Amazon Web Services
===================

``hailctl emr`` (experimental)
------------------------------

``hailctl emr`` provisions Amazon EMR on EC2 clusters configured for Hail. AWS support is
experimental and community-supported; it is not part of Hail's required continuous-integration
matrix.

The supported release is ``emr-spark-8.1.0`` (Spark 4.1.x and Scala 2.13). Hail and the bundled JAR
must be built from the same source revision for Spark 4.1. The command validates this information
from an artifact manifest before creating a cluster.

Artifact manifest
~~~~~~~~~~~~~~~~~

Build and upload an exact Hail wheel and an offline dependency wheelhouse. The local manifest passed
to ``hailctl`` records their S3 locations and SHA256 digests:

.. code-block:: json

    {
      "schema_version": 1,
      "hail_git_revision": "0123456789abcdef0123456789abcdef01234567",
      "hail_pip_version": "0.2.140",
      "spark_version": "4.1.2",
      "scala_version": "2.13.18",
      "python_version": "3.12",
      "wheel_uri": "s3://my-artifacts/hail.whl",
      "wheel_sha256": "<64 lowercase hex characters>",
      "wheelhouse_uri": "s3://my-artifacts/wheelhouse.tar.gz",
      "wheelhouse_sha256": "<64 lowercase hex characters>"
    }

The wheelhouse archive must contain ``requirements.txt`` and every dependency wheel needed by Hail,
excluding PySpark. EMR supplies PySpark. Bootstrap downloads both artifacts through the instance
profile, verifies their checksums, and installs without contacting PyPI.

Starting and using a cluster
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: text

    hailctl emr start CLUSTER_NAME \
        --artifact-manifest ./hail-emr-artifact.json \
        --s3-scratch s3://my-bucket/hail-tmp/ \
        --subnet-id subnet-PRIVATE \
        --service-access-security-group sg-EMR_SERVICE_ENDPOINT

Clusters receive an idle auto-termination policy. The default is one hour and can be changed with
``--idle-timeout``.

Submit a Python job and wait for it to finish:

.. code-block:: text

    hailctl emr submit CLUSTER_ID SCRIPT.py \
        --s3-scratch s3://my-bucket/hail-tmp/ [-- args to your script...]

List or terminate clusters:

.. code-block:: text

    hailctl emr list
    hailctl emr stop CLUSTER_ID

Runtime environment
~~~~~~~~~~~~~~~~~~~

Amazon EMR Spark 8 uses S3A rather than EMRFS. Hail routes ``s3://`` and ``s3a://`` paths through
Spark's Hadoop configuration. Hail sets ``HAIL_CLOUD=aws`` to select this HadoopFS-only route.

Hail requires Python 3.12 or later. EMR on EC2 does not preinstall Python 3.12, so the bootstrap uses
Amazon Linux 2023 ``dnf`` packages and points ``spark.pyspark.python`` and ``PYSPARK_PYTHON`` at
``/usr/bin/python3.12``.

Private subnet prerequisites
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Private clusters require network and IAM resources prepared outside ``hailctl``:

* an EMR service VPC endpoint or service-role permission to create and modify it;
* a service access security group with the EMR Spark 8 HTTPS rules;
* an S3 gateway endpoint or another controlled S3 route;
* endpoint policies that permit EMR instance-data buckets and the Hail artifact, log, scratch, and
  dataset buckets;
* a service role and EC2 instance profile with least-privilege S3, EC2, EMR, and optional KMS
  permissions.

This initial release requires a private subnet with controlled NAT egress because the bootstrap installs
Python 3.12 and native system packages from Amazon Linux repositories. Hail Python dependencies are
installed from the checksum-verified S3 wheelhouse rather than PyPI. A no-NAT deployment requires a
custom image or private package-repository design and is not currently supported.

Advanced cluster options
~~~~~~~~~~~~~~~~~~~~~~~~

``--run-job-flow-json`` deep-merges a JSON object into the final ``RunJobFlow`` request. Nested
objects are merged and lists are replaced. ``InstanceFleets`` replaces default ``InstanceGroups``.
The final request must retain Spark, the content-addressed Hail bootstrap, an S3 log URI, a supported
release, and a valid auto-termination policy.

``--off-heap-memory-per-core-mb`` caps Hail's native off-heap allocation per task core. It does not
reserve YARN container memory or automatically change ``spark.executor.memoryOverhead``.

Variant Effect Predictor (VEP)
------------------------------

VEP on EMR is not supported in this initial Spark 4 release.
