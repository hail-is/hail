.. _sec-getting_started:

===============
Getting Started
===============

You'll need:

- The `Java 8 JDK <http://www.oracle.com/technetwork/java/javase/downloads/index.html>`_.
- `Spark 2.0.2 <http://spark.apache.org/downloads.html>`_. Hail should work with other versions of Spark 2, see below.
- Python 2.7 and Jupyter Notebooks. We recommend the free `Anaconda distribution <https://www.continuum.io/downloads>`_.

-----------------------------------------------------
Running Hail locally with a pre-compiled distribution
-----------------------------------------------------

.. include:: distLinks.rst

Unzip the distribution after you download it. Next, edit and copy the below bash commands to set up the Hail
environment variables. You may want to add these to the appropriate dot-file (we recommend ``~/.profile``)
so that you don't need to rerun these commands in each new session.

Here, fill in the path to the **un-tarred** Spark package.

.. code-block:: text

    export SPARK_HOME=???

Here, fill in the path to the **unzipped** Hail distribution.

.. code-block:: text

    export HAIL_HOME=???
    export PATH=$PATH:$HAIL_HOME/bin/
    
Once you've set up Hail, we recommend that you run the Python tutorials to get an overview of Hail
functionality and learn about the powerful query language. To try Hail out, run the below commands
to start a Jupyter Notebook server in the tutorials directory.

.. code-block:: text

    cd $HAIL_HOME/tutorials
    jhail

You can now click on the "hail-overview" notebook to get started!

Running on a Spark cluster
==========================

Hail can run on any cluster that has Spark 2 installed. The Hail team publishes
ready-to-use JARs for Google Cloud Dataproc, see
:ref:`running-in-the-cloud`. For Cloudera specific instructions see
:ref:`running-on-a-cloudera-cluster`.

For all other Spark clusters, you will need to build Hail from the source code.

Hail should be built on the master node of the Spark cluster with the following
command, replacing ``2.0.2`` with the version of Spark available on your
cluster::

    ./gradlew -Dspark.version=2.0.2 shadowJar archiveZip

An IPython shell which can run Hail backed by the cluster can be started with
the following command, it is important that the Spark located at ``SPARK_HOME``
has the exact same version as provided to the previous command::

    SPARK_HOME=/path/to/spark/ \
    HAIL_HOME=/path/to/hail/ \
    PYTHONPATH="$PYTHONPATH:$HAIL_HOME/build/distributions/hail-python.zip:$SPARK_HOME/python:$SPARK_HOME/python/lib/py4j-*-src.zip" \
    ipython

Within the interactive shell, check that you can create a
``HailContext`` by running the following commands. Note that you have to pass in
the existing ``SparkContext`` instance ``sc`` to the ``HailContext``
constructor.

  .. code-block:: python

    >>> from hail import *
    >>> hc = HailContext(sc)
    
Files can be accessed from both Hadoop and Google Storage. If you're running on Google's Dataproc, you'll want to store your files in Google Storage. In most on premises clusters, you'll want to store your files in Hadoop.

To convert *sample.vcf* stored in Google Storage into Hail's **.vds** format, run:

  .. code-block:: python

    >>> hc.import_vcf('gs:///path/to/sample.vcf').write('gs:///output/path/sample.vds')
    
To convert *sample.vcf* stored in Hadoop into Hail's **.vds** format, run:

   .. code-block:: python

    >>> hc.import_vcf('/path/to/sample.vcf').write('/output/path/sample.vds')

It is also possible to run Hail non-interactively, by passing a Python script to
``spark-submit``. In this case, it is not necessary to set any environment
variables.

For example,

.. code-block:: text

    spark-submit --jars build/libs/hail-all-spark.jar \
                 --py-files build/distributions/hail-python.zip \
                 hailscript.py

runs the script `hailscript.py` (which reads and writes files from Hadoop):

.. code-block:: python

    import hail
    hc = hail.HailContext()
    hc.import_vcf('/path/to/sample.vcf').write('/output/path/sample.vds')

.. _running-on-a-cloudera-cluster:

Running on a Cloudera Cluster
=============================

`These instructions
<https://www.cloudera.com/documentation/spark2/latest/topics/spark2_installing.html>`_
explain how to install Spark 2 on a Cloudera cluster. You should work on a
gateway node on the cluster that has the Hadoop and Spark packages installed on
it.

Once Spark is installed, building and running Hail on a Cloudera cluster is exactly
the same as above, except:

 - On a Cloudera cluster, when building a Hail JAR, you must specify a Cloudera
   version of Spark. The following example builds a Hail JAR for Cloudera's
   2.0.2 version of Spark::
 
    ./gradlew shadowJar -Dspark.version=2.0.2.cloudera

 - On a Cloudera cluster, ``SPARK_HOME`` should be set as:
   ``SPARK_HOME=/opt/cloudera/parcels/SPARK2/lib/spark2``,

 - On Cloudera, you can create an interactive Python shell using ``pyspark2``::
 
    pyspark2 --jars build/libs/hail-all-spark.jar \
             --py-files build/distributions/hail-python.zip \
             --conf spark.sql.files.openCostInBytes=1099511627776 \
             --conf spark.sql.files.maxPartitionBytes=1099511627776 \
             --conf spark.kryo.registrator=is.hail.kryo.HailKryoRegistrator \
             --conf spark.hadoop.parquet.block.size=1099511627776

 - Cloudera's version of ``spark-submit`` is called ``spark2-submit``.

.. _running-in-the-cloud:

Running in the cloud
====================

`Google <https://cloud.google.com/dataproc/>`_ and `Amazon
<https://aws.amazon.com/emr/details/spark/>`_ offer optimized Spark performance
and exceptional scalability to tens of thousands of cores without the overhead
of installing and managing an on-prem cluster.

Hail publishes pre-built JARs for Google Cloud Platform's Dataproc Spark
clusters. We recommend running Hail on GCP via an interactive Jupyter notebook,
as described in `Liam's forum post
<http://discuss.hail.is/t/using-hail-with-jupyter-notebooks-on-google-cloud/196>`__. If
you prefer to submit your own JARs or python files rather than use a Jupyter
notebook, see `Laurent's forum post
<http://discuss.hail.is/t/using-hail-on-the-google-cloud-platform/80>`__.

Building with other versions of Spark 2
=======================================

Hail should work with other versions of Spark 2.  To build against a
different version, such as Spark 2.1.0, modify the above
instructions as follows:

 - Set the Spark version in the gradle command

   .. code-block:: text

      ./gradlew -Dspark.version=2.1.0 shadowJar

 - ``SPARK_HOME`` should point to an installation of the desired version of Spark, such as *spark-2.1.0-bin-hadoop2.7*

 - The version of the Py4J ZIP file in the hail alias must match the version in ``$SPARK_HOME/python/lib`` in your version of Spark.

---------------
BLAS and LAPACK
---------------

Hail uses BLAS and LAPACK optimized linear algebra libraries. These should load automatically on recent versions of Mac OS X and Google Dataproc. On Linux, these must be explicitly installed; on Ubuntu 14.04, run

.. code-block:: text

    apt-get install libatlas-base-dev

If natives are not found, ``hail.log`` will contain the warnings

.. code-block:: text

    Failed to load implementation from: com.github.fommil.netlib.NativeSystemLAPACK
    Failed to load implementation from: com.github.fommil.netlib.NativeSystemBLAS

See `netlib-java <http://github.com/fommil/netlib-java>`_ for more information.
