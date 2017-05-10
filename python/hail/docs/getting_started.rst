.. _sec-getting_started:

.. testsetup::

    hc.stop()

===============
Getting Started
===============

You'll need:

- The `Java 8 JDK <http://www.oracle.com/technetwork/java/javase/downloads/index.html>`_.
- `Spark 2.0.2 <http://spark.apache.org/downloads.html>`_. Hail should work with other versions of Spark 2, see below.
- Python 2.7 and IPython. We recommend the free `Anaconda distribution <https://www.continuum.io/downloads>`_.
- `CMake <http://cmake.org>`_ and a C++ compiler that supports ``-std=c++11`` (we recommend at least GCC 4.7 or Clang 3.3).

  On a Debian-based Linux OS like Ubuntu, run:

  .. code-block:: text

    $ sudo apt-get install g++ cmake

  On Mac OS X, install Xcode, available through the App Store, for the C++ compiler.  `CMake <http://cmake.org>`_ can be downloaded from the CMake website or through `Homebrew <http://brew.sh>`_.  To install with Homebrew, run

  .. code-block:: text

    $ brew install cmake

- The Hail source code.  To clone the `Hail repository <https://github.com/broadinstitute/hail>`_ using `Git <https://git-scm.com/>`_, run

  .. code-block:: text

      $ git clone https://github.com/broadinstitute/hail.git
      $ cd hail

  You can also download the source code directly from `Github <https://github.com/broadinstitute/hail/archive/master.zip>`_.

  You may also want to install `Seaborn <http://seaborn.pydata.org>`_, a Python library for statistical data visualization, using ``conda install seaborn`` or ``pip install seaborn``. While not technically necessary, Seaborn is used in the tutorial to make prettier plots.

To install all dependencies for running locally on a fresh Ubuntu installation, use this `script <https://github.com/hail-is/hail/wiki/Install-Hail-dependencies-on-a-fresh-Ubuntu-VM>`_.

The following commands are relative to the ``hail`` directory.

-------------------------
Building and running Hail
-------------------------

Hail may be built to run locally or on a Spark cluster. Running locally is useful for getting started, analyzing or experimenting with small datasets, and Hail development.

Running locally
===============

The single command

  .. code-block:: text

    $ ./gradlew shadowJar

creates a Hail JAR file at ``build/libs/hail-all-spark.jar``. The initial build takes time as `Gradle <https://gradle.org/>`_ installs all Hail dependencies.

Add the following environmental variables by filling in the paths to **SPARK_HOME** and **HAIL_HOME** below and exporting all four of them (consider adding them to your .bashrc)::

    $ export SPARK_HOME=/path/to/spark
    $ export HAIL_HOME=/path/to/hail
    $ export PYTHONPATH="$PYTHONPATH:$HAIL_HOME/python:$SPARK_HOME/python:`echo $SPARK_HOME/python/lib/py4j*-src.zip`"
    $ export SPARK_CLASSPATH=$HAIL_HOME/build/libs/hail-all-spark.jar

Running ``ipython`` on the command line will open an interactive Python shell.

Here are a few simple things to try in order. To import the ``hail`` module and start a :py:class:`~hail.HailContext`, run:

.. doctest::

    >>> from hail import *
    >>> hc = HailContext()

To :func:`import <hail.HailContext.import_vcf>` the included *sample.vcf* into Hail's **.vds** format, run:

.. doctest::
    :hide:

    >>> hc.import_vcf('../../../src/test/resources/sample.vcf').write('sample.vds')

.. doctest::
    :options: +SKIP

    >>> hc.import_vcf('src/test/resources/sample.vcf').write('sample.vds')

To :func:`split <hail.VariantDataset.split_multi>` multi-allelic variants, compute a panel of :func:`sample <hail.VariantDataset.sample_qc>` and :func:`variant <hail.VariantDataset.sample_qc>` quality control statistics, write these statistics to files, and save an annotated version of the vds, run:

.. doctest::

    >>> vds = (hc.read('sample.vds')
    ...     .split_multi()
    ...     .sample_qc()
    ...     .variant_qc())
    >>> vds.export_variants('variantqc.tsv', 'Variant = v, va.qc.*')
    >>> vds.write('sample.qc.vds')


To :func:`count <hail.VariantDataset.count>` the number of samples and variants, run:

.. doctest::

    >>> vds.count()

The :func:`summarize <hail.VariantDataset.summarize>` method produces a variety of useful statistics:

.. doctest::

    >>> vds.summarize().report()

Now let's get a feel for Hail's powerful objects, annotation system, and `expression language <exprlang>`_. To print the current annotation schema and use these annotations to filter variants, samples, and genotypes, run:

.. doctest::

    >>> print('sample annotation schema:')
    >>> print(vds.sample_schema)
    >>> print('\nvariant annotation schema:')
    >>> print(vds.variant_schema)
    >>> (vds.filter_variants_expr('v.altAllele().isSNP() && va.qc.gqMean >= 20')
    ...     .filter_samples_expr('sa.qc.callRate >= 0.97 && sa.qc.dpMean >= 15')
    ...     .filter_genotypes('let ab = g.ad[1] / g.ad.sum() in '
    ...                       '((g.isHomRef() && ab <= 0.1) || '
    ...                       ' (g.isHet() && ab >= 0.25 && ab <= 0.75) || '
    ...                       ' (g.isHomVar() && ab >= 0.9))')
    ...     .write('sample.filtered.vds'))

Try running :py:meth:`~hail.VariantDataset.count` on *sample.filtered.vds* to see how the numbers have changed. For further background and examples, continue to the :ref:`sec-overview` and :ref:`API reference <sec-api>`.

Note that during each run Hail writes a ``hail.log`` file in the current directory; this is useful to developers for debugging.

Running on a Spark cluster
==========================

Hail can run on any cluster that has Spark 2 installed. For instructions
specific to Google Cloud Dataproc clusters and Cloudera clusters, see below.

For all other Spark clusters, you will need to build Hail from the source
code. First, log onto the master node of the Spark cluster, and build a Hail JAR
and a zipfile of the Python code by running:

  .. code-block:: text

    $ ./gradlew shadowJar archiveZip

Then set the following environment variables with values peculiar to your
cluster for ``SPARK_HOME`` and ``HAIL_HOME``::

    $ export SPARK_HOME=/path/to/spark/
    $ export HAIL_HOME=/path/to/hail
    $ export PYTHONPATH="$PYTHONPATH:$HAIL_HOME/build/distributions/hail-python.zip:$SPARK_HOME/python:$SPARK_HOME/python/lib/py4j-*-src.zip"

You can open an IPython shell with the ``ipython`` command:

  .. code-block:: text

    $ ipython

Within the interactive shell, check that you can successfully create a
``HailContext`` by running the following commands. Note that you have to pass in
the existing ``SparkContext`` instance ``sc`` to the ``HailContext``
constructor.

  .. code-block:: python

    >>> import hail
    >>> hc = hail.HailContext(sc)
    
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

    $ spark-submit --jars build/libs/hail-all-spark.jar \
                   --py-files build/distributions/hail-python.zip \
                   hailscript.py

runs the script `hailscript.py` (which reads and writes files from Hadoop):

  .. code-block:: python

    import hail
    hc = hail.HailContext()
    hc.import_vcf('/path/to/sample.vcf').write('/output/path/sample.vds')

Running on a Cloudera Cluster
=============================

`These instructions
<https://www.cloudera.com/documentation/spark2/latest/topics/spark2_installing.html>`_
explain how to install Spark 2 on a Cloudera cluster. You should work on a
gateway node on the cluster that has the Hadoop and Spark packages installed on
it.

Once Spark is installed, running Hail on a Cloudera cluster is exactly the same
as above, except:

 - On a Cloudera cluster, ``SPARK_HOME`` should be set as:
   ``SPARK_HOME=/opt/cloudera/parcels/SPARK2/lib/spark2``,

 - On Cloudera, you can create an interactive Python shell using ``pyspark2``:
 
     $ pyspark2 --jars build/libs/hail-all-spark.jar \
              --py-files build/distributions/hail-python.zip \
              --conf spark.hadoop.io.compression.codecs=org.apache.hadoop.io.compress.DefaultCodec,is.hail.io.compress.BGzipCodec,org.apache.hadoop.io.compress.GzipCodec \
              --conf spark.sql.files.openCostInBytes=1099511627776 \
              --conf spark.sql.files.maxPartitionBytes=1099511627776 \
              --conf spark.hadoop.mapreduce.input.fileinputformat.split.minsize=1099511627776 \
              --conf spark.hadoop.parquet.block.size=1099511627776

 - Cloudera's version of ``spark-submit`` is called ``spark2-submit``.

Running in the cloud
====================

`Google <https://cloud.google.com/dataproc/>`_ and `Amazon
<https://aws.amazon.com/emr/details/spark/>`_ offer optimized Spark performance
and exceptional scalability to tens of thousands of cores without the overhead
of installing and managing an on-prem cluster.

Hail publishes pre-built JARs for Google Cloud Platform's Dataproc Spark
clusters. If you would prefer to avoid building Hail from source, learn how to
get started on Google Cloud Platform by reading this `forum post
<http://discuss.hail.is/t/using-hail-on-the-google-cloud-platform/80>`__. To get
started running Hail on GCP via an interactive Jupyter notebook, see this `forum post
<http://discuss.hail.is/t/using-hail-with-jupyter-notebooks-on-google-cloud/196>`__.

Building with other versions of Spark 2
=======================================

Hail should work with other versions of Spark 2.  To build against a
different version, such as Spark 2.1.0, modify the above
instructions as follows:

 - Set the Spark version in the gradle command

   .. code-block:: text

      $ ./gradlew -Dspark.version=2.1.0 shadowJar

 - ``SPARK_HOME`` should point to an installation of the desired version of Spark, such as *spark-2.1.0-bin-hadoop2.7*

 - The version of the Py4J ZIP file in the hail alias must match the version in ``$SPARK_HOME/python/lib`` in your version of Spark.

---------------
BLAS and LAPACK
---------------

Hail uses BLAS and LAPACK optimized linear algebra libraries. These should load automatically on recent versions of Mac OS X and Google Dataproc. On Linux, these must be explicitly installed; on Ubuntu 14.04, run

.. code-block:: text

    $ apt-get install libatlas-base-dev

If natives are not found, ``hail.log`` will contain the warnings

.. code-block:: text

    Failed to load implementation from: com.github.fommil.netlib.NativeSystemLAPACK
    Failed to load implementation from: com.github.fommil.netlib.NativeSystemBLAS

See `netlib-java <http://github.com/fommil/netlib-java>`_ for more information.

-----------------
Running the tests
-----------------

Several Hail tests have additional dependencies:

 - `PLINK 1.9 <http://www.cog-genomics.org/plink2>`_

 - `QCTOOL 1.4 <http://www.well.ox.ac.uk/~gav/qctool>`_

 - `R 3.3.1 <http://www.r-project.org/>`_ with packages ``jsonlite`` and ``logistf``, which depends on ``mice`` and ``Rcpp``.

Other recent versions of QCTOOL and R should suffice, but PLINK 1.7 will not.

To execute all Hail tests, run

.. code-block:: text

    $ ./gradlew -Dspark.home=$SPARK_HOME test
