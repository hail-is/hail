.. _sec-getting_started:

===============
Getting Started
===============

You'll need:

- The `Java 8 JDK <http://www.oracle.com/technetwork/java/javase/downloads/index.html>`_.
- `Spark 2.0.2 <http://spark.apache.org/downloads.html>`_. Hail should work with other versions of Spark 2, see below.
- Python 2.7 and IPython. We recommend the free `Anaconda distribution <https://www.continuum.io/downloads>`_.
- `CMake <http://cmake.org>`_ and a C++ compiler.

  On a Debian-based Linux like Ubuntu, run:

  .. code-block:: text

    $ sudo apt-get install g++ cmake

  On OSX, install Xcode, available through the App Store, for the C++ compiler.  `CMake <http://cmake.org>`_ can be downloaded from the CMake website or through `Homebrew <http://brew.sh>`_.  To install with Homebrew, run

  .. code-block:: text

    $ brew install cmake

- The Hail source code.  To clone the `Hail repository <https://github.com/broadinstitute/hail>`_ using `Git <https://git-scm.com/>`_, run

  .. code-block:: text

      $ git clone https://github.com/broadinstitute/hail.git
      $ cd hail

  You can also download the source code directly from `Github <https://github.com/broadinstitute/hail/archive/master.zip>`_.

  Please also install "Seaborn", a python library for statistical data visualization. The easiest way to do this is by running:

  .. code-block:: text

      $ pip install seaborn

  or 

  .. code-block:: text

      $ conda install seaborn

  The former will only work if you pip installed, the later will only work if you did install Anaconda above. Seaborn is not technically 
  necessary, but it helps with the graphs in the tutorial.

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

Add the following environmental variables and make an alias for Hail by filling in the paths to **SPARK_HOME** and **HAIL_HOME** below::

    $ SPARK_HOME=/path/to/spark
    $ HAIL_HOME=/path/to/hail
    $ alias hail="PYTHONPATH=$SPARK_HOME/python:$SPARK_HOME/python/lib/py4j-0.10.3-src.zip:$HAIL_HOME/python SPARK_CLASSPATH=$HAIL_HOME/build/libs/hail-all-spark.jar python"

Running ``hail`` on the command line will open an interactive Python shell.

Here are a few simple things to try in order. To import the ``hail`` module and start a :py:class:`~hail.HailContext`, run::

    >>> import hail
    >>> hc = hail.HailContext()

To :func:`import <hail.HailContext.import_vcf>` the included *sample.vcf* into Hail's **.vds** format, run::

    >>> hc.import_vcf('src/test/resources/sample.vcf').write('sample.vds')

To :func:`split <hail.VariantDataset.split_multi>` multi-allelic variants, compute a panel of :func:`sample <hail.VariantDataset.sample_qc>` and :func:`variant <hail.VariantDataset.sample_qc>` quality control statistics, write these statistics to files, and save an annotated version of the vds, run::

    >>> vds = (hc.read('sample.vds')
    >>>     .split_multi()
    >>>     .sample_qc()
    >>>     .variant_qc()
    >>>     .export_variants('variantqc.tsv', 'Variant = v, va.qc.*')
    >>>     .write('sample.qc.vds'))


To :func:`count <hail.VariantDataset.count>` the number of samples, variants, and genotypes, run::

    >>> vds.count(genotypes=True)

Now let's get a feel for Hail's powerful :ref:`objects <sec-objects>`, `annotation system <../reference.html#Annotations>`_, and `expression language <../reference.html#HailExpressionLanguage>`_. To :func:`print <hail.VariantDataset.print_schema>` the current annotation schema and use these annotations to filter variants, samples, and genotypes, run::

    >>> (vds.print_schema('schema.txt')
    >>>     .filter_variants_expr('v.altAllele.isSNP && va.qc.gqMean >= 20')
    >>>     .filter_samples_expr('sa.qc.callRate >= 0.97 && sa.qc.dpMean >= 15')
    >>>     .filter_genotypes('let ab = g.ad[1] / g.ad.sum in '
    >>>                       '((g.isHomRef && ab <= 0.1) || '
    >>>                       ' (g.isHet && ab >= 0.25 && ab <= 0.75) || '
    >>>                       ' (g.isHomVar && ab >= 0.9))')
    >>>     .write('sample.filtered.vds'))

Try running :py:meth:`~hail.VariantDataset.count` on *sample.filtered.vds* to see how the numbers have changed. For further background and examples, continue to the :ref:`sec-overview` and :ref:`API reference <sec-api>`.

Note that during each run Hail writes a ``hail.log`` file in the current directory; this is useful to developers for debugging.

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

Running on a Spark cluster and in the cloud
===========================================

The ``build/libs/hail-all-spark.jar`` can be submitted using ``spark-submit``. See the `Spark documentation <http://spark.apache.org/docs/latest/cluster-overview.html>`_ for details.

`Google <https://cloud.google.com/dataproc/>`_ and `Amazon <https://aws.amazon.com/emr/details/spark/>`_ offer optimized Spark performance and exceptional scalability to tens of thousands of cores without the overhead of installing and managing an on-prem cluster.
To get started running Hail on the Google Cloud Platform, see this `forum post <http://discuss.hail.is/t/using-hail-on-the-google-cloud-platform/80>`_.

---------------
BLAS and LAPACK
---------------

Hail uses BLAS and LAPACK optimized linear algebra libraries. On Linux, these must be explicitly installed. On Ubuntu 14.04, run

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