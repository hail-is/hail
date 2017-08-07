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
environment variables. You may want to add these to your bash dot-file (``~/.bash_profile``, ``~/.bashrc``, etc)
so that you don't need to rerun these commands in each new session.

Here, fill in the path to the un-tarred Spark package.

.. code-block:: text

    export SPARK_HOME=???

Here, fill in the path to the unzipped Hail distribution.

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

-------------------------
Building Hail from source
-------------------------

A Hail JAR and python library can also be built from the source; however, we don't recommend this for most users.

The minimal set of tools necessary to build Hail from source are a C++ compiler and CMake. On a Debian-based OS like Ubuntu, these tools can be installed with apt-get::

    sudo apt-get install g++ cmake

On Mac OS X, a C++ compiler is provided by the Apple Xcode, which the following command installs::

    xcode-select --install

The recommended way to install CMake is with `Homebrew <http://brew.sh>`_::

    brew install cmake

The Hail source code is hosted `on GitHub <https://github.com/broadinstitute/hail>`_::

    git clone https://github.com/broadinstitute/hail.git
    cd hail

You may also want to install `Seaborn <http://seaborn.pydata.org>`_, a Python library for statistical data visualization, using ``conda install seaborn`` or ``pip install seaborn``. While not technically necessary, Seaborn is used in the tutorials to make prettier plots.

A Hail JAR can be built using Gradle::

    ./gradlew -Dspark.version=2.0.2 shadowJar

Finally, some environment variables must be set so that Hail can find Spark, Spark can find Hail, and Python can find Hail. Add these lines to your ``.bashrc`` or equivalent setting ``SPARK_HOME`` to the root directory of a Spark installation and ``HAIL_HOME`` to the root of the Hail repository::

    export SPARK_HOME=/path/to/spark
    export HAIL_HOME=/path/to/hail
    export PYTHONPATH="$PYTHONPATH:$HAIL_HOME/python:$SPARK_HOME/python:`echo $SPARK_HOME/python/lib/py4j*-src.zip`"
    export SPARK_CLASSPATH=$HAIL_HOME/build/libs/hail-all-spark.jar

Now you can import hail from a python interpreter::

    # python
    Python 2.7.12 |Anaconda custom (x86_64)| (default, Jul  2 2016, 17:43:17) 
    [GCC 4.2.1 (Based on Apple Inc. build 5658) (LLVM build 2336.11.00)] on darwin
    Type "help", "copyright", "credits" or "license" for more information.
    Anaconda is brought to you by Continuum Analytics.
    Please check out: http://continuum.io/thanks and https://anaconda.org
    >>> from hail import *
    >>> hc = HailContext()
    Using Spark's default log4j profile: org/apache/spark/log4j-defaults.properties
    Setting default log level to "WARN".
    To adjust logging level use sc.setLogLevel(newLevel).
    hail: info: SparkUI: http://10.1.1.163:4040
    Welcome to
         __  __     <>__
        / /_/ /__  __/ /
       / __  / _ `/ / /
      /_/ /_/\_,_/_/_/   version devel-b2394a4
    >>> 


Running on a Spark cluster
==========================

Hail can run on any cluster that has Spark 2 installed. For instructions
specific to Google Cloud Dataproc clusters and Cloudera clusters, see below.

For all other Spark clusters, you will need to build Hail from the source code.

Hail should be built on the master node of the Spark cluster with the following command::

    ./gradlew -Dspark.version=2.0.2 shadowJar archiveZip

An IPython shell which can run Hail backed by the cluster can be started by::

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

    ./gradlew -Dspark.home=$SPARK_HOME test
