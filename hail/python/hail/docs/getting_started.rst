.. _sec-installation:

===============
Installing Hail
===============

Requirements
------------

Regardless of installation method, you will need:

- `Java 8 JDK
  <http://www.oracle.com/technetwork/java/javase/downloads/jdk8-downloads-2133151.html>`_
  Note: it *must* be version eight. Hail does not support Java versions nine,
  ten, or eleven due to our dependency on Spark.
- Python 3.6 or later, we recommend `Anaconda's Python 3
  <https://www.anaconda.com/download/>`_

For all methods *other than using pip*, you will additionally need `Spark
2.2.x
<https://www.apache.org/dyn/closer.lua/spark/spark-2.2.2/spark-2.2.2-bin-hadoop2.7.tgz>`_.


Installation
------------

Installing Hail on Mac OS X or GNU/Linux with pip
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If you have Mac OS X, this is the recommended installation method for running
Hail locally (i.e. not on a cluster).

Create a `conda enviroment
<https://conda.io/docs/user-guide/concepts.html#conda-environments>`__ named
``hail`` and install the Hail python library in that environment:

.. code-block:: sh

    conda create --name hail python>=3.6
    conda activate hail
    pip install hail

To verify installation, open iPython or a Jupyter notebook and run:

.. code-block:: python

    >>> import hail as hl
    >>> mt = hl.balding_nichols_model(n_populations=3, n_samples=50, n_variants=100)
    >>> mt._force_count()

You're now all set to run the
`tutorials <https://hail.is/docs/devel/tutorials-landing.html>`__ locally!

Building your own Jar
~~~~~~~~~~~~~~~~~~~~~

To use Hail with other Hail versions of Spark 2, you'll need to build your own JAR instead of using a pre-compiled
distribution. To build against a different version, such as Spark 2.3.0, run the following command inside the directory
where Hail is located::

    ./gradlew -Dspark.version=2.3.0 shadowJar

The Spark version in this command should match whichever version of Spark you would like to build against.

The ``SPARK_HOME`` environment variable should point to an installation of the desired version of Spark, such as *spark-2.3.0-bin-hadoop2.7*

The version of the Py4J ZIP file in the hail alias must match the version in ``$SPARK_HOME/python/lib`` in your version of Spark.

Running on a Spark cluster
~~~~~~~~~~~~~~~~~~~~~~~~~~

Hail can run on any Spark 2.2 cluster. For example, Google and Amazon offer
optimized Spark performance and exceptional scalability to thousands of cores
without the overhead of installing and managing an on-premesis cluster.

On `Google Dataproc <https://cloud.google.com/dataproc/>`_,
we provide pre-built JARs and a Python package
`cloudtools <https://github.com/Nealelab/cloudtools>`_
to simplify running Hail, whether through an interactive Jupyter notebook or by
submitting Python scripts.

On `Amazon EMR <https://aws.amazon.com/emr/details/spark/>`_, we recommend using the Hail
`cloudformation <https://github.com/hms-dbmi/hail-on-AWS-spot-instances>`_ tool
developed by Carlos De Niz in the
`Avillach Lab <https://avillach-lab.hms.harvard.edu/>`_ at Harvard Medical School.

For Cloudera-specific instructions, see :ref:`running-on-a-cloudera-cluster`.

For all other Spark clusters, you will need to build Hail from the source code.

Hail should be built on the master node of the Spark cluster with the following
command, replacing ``2.2.0`` with the version of Spark available on your
cluster::

    ./gradlew -Dspark.version=2.2.0 shadowJar archiveZip

Python and IPython need a few environment variables to correctly find Spark and
the Hail jar. We recommend you set these environment variables in the relevant
profile file for your shell (e.g. ``~/.bash_profile``).

.. code-block:: sh

    export SPARK_HOME=/path/to/spark-2.2.0/
    export HAIL_HOME=/path/to/hail/
    export PYTHONPATH="${PYTHONPATH:+$PYTHONPATH:}$HAIL_HOME/build/distributions/hail-python.zip"
    export PYTHONPATH="$PYTHONPATH:$SPARK_HOME/python"
    export PYTHONPATH="$PYTHONPATH:$SPARK_HOME/python/lib/py4j-*-src.zip"
    ## PYSPARK_SUBMIT_ARGS is used by ipython and jupyter
    export PYSPARK_SUBMIT_ARGS="\
      --jars $HAIL_HOME/build/libs/hail-all-spark.jar \
      --conf spark.driver.extraClassPath=\"$HAIL_HOME/build/libs/hail-all-spark.jar\" \
      --conf spark.executor.extraClassPath=./hail-all-spark.jar \
      --conf spark.serializer=org.apache.spark.serializer.KryoSerializer \
      --conf spark.kryo.registrator=is.hail.kryo.HailKryoRegistrator
      pyspark-shell"

If the previous environment variables are set correctly, an IPython shell which
can run Hail backed by the cluster can be started with the following command::

    ipython

When using ``ipython``, you can import hail and start interacting directly:

.. code-block:: python

    >>> import hail as hl
    >>> mt = hl.balding_nichols_model(n_populations=3, n_samples=50, n_variants=100)
    >>> mt._force_count()

You can also interact with hail via a ``pyspark`` session, but you will need to
pass the configuration from ``PYSPARK_SUBMIT_ARGS`` directly as well as adding
extra configuration parameters specific to running Hail through ``pyspark``::

    pyspark \
      --jars $HAIL_HOME/build/libs/hail-all-spark.jar \
      --conf spark.driver.extraClassPath=$HAIL_HOME/build/libs/hail-all-spark.jar \
      --conf spark.executor.extraClassPath=./hail-all-spark.jar \
      --conf spark.serializer=org.apache.spark.serializer.KryoSerializer \
      --conf spark.kryo.registrator=is.hail.kryo.HailKryoRegistrator

Moreover, unlike in ``ipython``, ``pyspark`` provides a Spark Context via the
global variable ``sc``. For Hail to interact properly with the Spark cluster,
you must tell hail about this special Spark Context

.. code-block:: python

    >>> import hail as hl
    >>> hl.init(sc) # doctest: +SKIP

After this initialization step, you can interact as you would in ``ipython``

.. code-block:: python

    >>> mt = hl.balding_nichols_model(n_populations=3, n_samples=50, n_variants=100)
    >>> mt._force_count()

It is also possible to run Hail non-interactively, by passing a Python script to
``spark-submit``. Again, you will need to explicitly pass several configuration
parameters to ``spark-submit``::

    spark-submit \
      --jars "$HAIL_HOME/build/libs/hail-all-spark.jar" \
      --py-files "$HAIL_HOME/build/distributions/hail-python.zip" \
      --conf spark.driver.extraClassPath="$HAIL_HOME/build/libs/hail-all-spark.jar" \
      --conf spark.executor.extraClassPath=./hail-all-spark.jar \
      --conf spark.serializer=org.apache.spark.serializer.KryoSerializer \
      --conf spark.kryo.registrator=is.hail.kryo.HailKryoRegistrator \
      your-hail-python-script-here.py

.. _running-on-a-cloudera-cluster:

Running on a Cloudera cluster
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

`These instructions
<https://www.cloudera.com/documentation/spark2/latest/topics/spark2_installing.html>`_
explain how to install Spark 2 on a Cloudera cluster. You should work on a
gateway node on the cluster that has the Hadoop and Spark packages installed on
it.

Once Spark is installed, building and running Hail on a Cloudera cluster is exactly
the same as above, except:

 - On a Cloudera cluster, when building a Hail JAR, you must specify a Cloudera
   version of Spark. The following example builds a Hail JAR for Cloudera's
   2.2.0 version of Spark::

    ./gradlew shadowJar -Dspark.version=2.2.0.cloudera

 - On a Cloudera cluster, ``SPARK_HOME`` should be set as:
   ``SPARK_HOME=/opt/cloudera/parcels/SPARK2/lib/spark2``,

 - On Cloudera, you can create an interactive Python shell using ``pyspark``::

    pyspark --jars build/libs/hail-all-spark.jar \
            --py-files build/distributions/hail-python.zip \
            --conf spark.driver.extraClassPath="build/libs/hail-all-spark.jar" \
            --conf spark.executor.extraClassPath=./hail-all-spark.jar \
            --conf spark.serializer=org.apache.spark.serializer.KryoSerializer \
            --conf spark.kryo.registrator=is.hail.kryo.HailKryoRegistrator \


Common Installation Issues
--------------------------


BLAS and LAPACK
~~~~~~~~~~~~~~~

Hail uses BLAS and LAPACK optimized linear algebra libraries. These should load automatically on recent versions of Mac OS X and Google Dataproc. On Linux, these must be explicitly installed; on Ubuntu 14.04, run::

    apt-get install libatlas-base-dev

If natives are not found, ``hail.log`` will contain these warnings:

.. code-block:: text

    Failed to load implementation from: com.github.fommil.netlib.NativeSystemLAPACK
    Failed to load implementation from: com.github.fommil.netlib.NativeSystemBLAS

See `netlib-java <http://github.com/fommil/netlib-java>`_ for more information.

