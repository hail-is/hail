===================
Amazon Web Services
===================

``hailctl emr``
---------------

Pip installations of Hail come bundled with a command-line tool, ``hailctl emr``, for working with
`Amazon EMR <https://aws.amazon.com/emr/>`__ clusters configured for Hail.

This tool requires the `AWS credentials <https://docs.aws.amazon.com/cli/latest/userguide/cli-configure-files.html>`__
that boto3 uses (for example via ``aws configure``, environment variables, or an SSO profile). It
does not require the ``aws`` CLI to be installed. You must also have permission to create EMR
clusters and the EMR service and EC2 instance-profile IAM roles (``aws emr create-default-roles``
creates the defaults).

An EMR cluster consists of a master node and one or more core nodes. Hail is installed on every node
by a bootstrap action, and Hail's JAR is placed on the Spark classpath through the cluster's
configuration. Hail reads and writes ``s3://`` URIs directly through EMRFS.

To start a cluster you must supply a cluster name and an S3 scratch location (either ``--s3-scratch``
or the ``emr/remote_tmpdir`` config variable). The scratch location holds the uploaded bootstrap
script and, for ``submit``, your job scripts.

.. code-block:: text

    hailctl emr start CLUSTER_NAME --s3-scratch s3://my-bucket/hail-tmp/

To submit a Python job and wait for it to finish:

.. code-block:: text

    hailctl emr submit CLUSTER_ID SCRIPT.py --s3-scratch s3://my-bucket/hail-tmp/ [-- args to your script...]

Your script should read and write ``s3://`` URIs (EMRFS resolves them). To list running clusters:

.. code-block:: text

    hailctl emr list

Importantly, to shut down a cluster when you are done with it:

.. code-block:: text

    hailctl emr stop CLUSTER_ID

Choosing an EMR release
-----------------------

Hail requires Spark 3.5.x. The default release, ``emr-7.3.0``, provides Spark 3.5.3, which matches
the Spark version Hail is built against. If you pass a ``--release-label`` that ships a different
Spark minor version, ``hailctl emr start`` refuses to start the cluster; an unrecognized label
produces a warning.

Advanced cluster options
------------------------

``hailctl emr start`` exposes common options directly (instance types and counts, ``--ec2-key-name``,
``--subnet-id``, custom IAM roles). For any other EMR ``RunJobFlow`` setting, pass a JSON object with
``--run-job-flow-json``; it is deep-merged into the request, so you can set spot instance fleets,
extra applications, tags, and so on:

.. code-block:: text

    hailctl emr start CLUSTER_NAME --s3-scratch s3://my-bucket/hail-tmp/ \
        --run-job-flow-json '{"Tags": [{"Key": "team", "Value": "genomics"}]}'

Variant Effect Predictor (VEP)
------------------------------

Running VEP on EMR is not yet supported. Passing ``--vep`` raises an error. This is planned for a
future release.
