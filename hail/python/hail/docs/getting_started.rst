.. _sec-installation:

===============
Installing Hail
===============

Requirements
------------

Regardless of installation method, you will need:

- `Java 8 JDK
  <http://www.oracle.com/technetwork/java/javase/downloads/jdk8-downloads-2133151.html>`_
  Note: it *must* be Java **8**. Hail does not support versions 9+ due to our
  dependency on Spark.
- Python 3.6 or later. We recommend `Miniconda Python 3.7
  <https://docs.conda.io/en/latest/miniconda.html>`_

Regardless of installation method, GNU/Linux users must obtain a recent version
of the C and C++ standard library:

- GCC 5.0, and LLVM version 3.4 (which is Apple LLVM version 6.0) and later
  should install a compatible C++ standard library.
- Most GNU/Linux distributions released since 2012 have a compatible C standard
  library

For all methods *other than using pip*, you will additionally need

- `Spark 2.4.x <https://www.apache.org/dyn/closer.lua/spark/spark-2.4.0/spark-2.4.0-bin-hadoop2.7.tgz>`_,
- a C++ compiler, and
- lz4

For the latter two, Debian users might try::

    sudo apt-get install g++ liblz4-dev

and Mac OS X users, might try::

    xcode-select --install
    brew install lz4


Installation
------------

Installing Hail on Mac OS X or GNU/Linux with pip
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If you have Mac OS X, this is the recommended installation method for running
Hail locally (i.e. not on a cluster).

Create a `conda enviroment
<https://conda.io/docs/user-guide/concepts.html#conda-environments>`__ named
``hail`` and install the Hail python library in that environment. If ``conda activate`` doesn't work, `please read these instructions <https://conda.io/projects/conda/en/latest/user-guide/install/macos.html#install-macos-silent>`_

.. code-block:: sh

    conda create -n hail python==3.6
    conda activate hail
    pip3 install hail

To try Hail out, open iPython or a Jupyter notebook and run:

.. code-block:: python

    >>> import hail as hl
    >>> mt = hl.balding_nichols_model(n_populations=3, n_samples=50, n_variants=100)
    >>> mt.count()

You're now all set to run the
`tutorials <https://hail.is/docs/0.2/tutorials-landing.html>`__ locally!

Running on a Spark cluster
~~~~~~~~~~~~~~~~~~~~~~~~~~

Hail can run on any Spark 2.4 cluster. For example, Google and Amazon make it
possible to rent Spark clusters with many thousands of cores on-demand,
providing for the elastic compute requirements of scientific research without
an up-front capital investment.

For more about computing on the cloud, see `Hail on the cloud <hail_on_the_cloud.html>`_.

For Cloudera-specific instructions, see :ref:`running-on-a-cloudera-cluster`.

For all other Spark clusters, you will need to build Hail from the source code.

Hail should be built on the master node of the Spark cluster. The following
command builds Hail for Spark 2.4.0, installs the Python library, and installs
all the Python dependencies::

    make install-on-cluster HAIL_COMPILE_NATIVES=1 SPARK_VERSION=2.4.0

An IPython shell which can run Hail backed by the cluster can be started with
the following command::

    ipython

When using ``ipython``, you can import hail and start interacting directly:

.. code-block:: python

    >>> import hail as hl
    >>> mt = hl.balding_nichols_model(n_populations=3, n_samples=50, n_variants=100)
    >>> mt.count()

You can also interact with Hail via a ``pyspark`` session, but you will need to
configure the class path appropriately::

    HAIL_HOME=$(pip3 show hail | grep Location | awk -F' ' '{print $2 "/hail"}')
    pyspark \
      --jars $HAIL_HOME/hail-all-spark.jar \
      --conf spark.driver.extraClassPath=$HAIL_HOME/hail-all-spark.jar \
      --conf spark.executor.extraClassPath=./hail-all-spark.jar \
      --conf spark.serializer=org.apache.spark.serializer.KryoSerializer \
      --conf spark.kryo.registrator=is.hail.kryo.HailKryoRegistrator

Moreover, unlike in ``ipython``, ``pyspark`` provides a Spark Session via the
global variable ``spark``. For Hail to interact properly with the Spark cluster,
you must tell hail about this special Spark Session

.. code-block:: python

    >>> import hail as hl
    >>> hl.init(sc=spark.sparkContext) # doctest: +SKIP

After this initialization step, you can interact as you would in ``ipython``

.. code-block:: python

    >>> mt = hl.balding_nichols_model(n_populations=3, n_samples=50, n_variants=100)
    >>> mt.count()

It is also possible to run Hail non-interactively, by passing a Python script to
``spark-submit``. Again, you will need to explicitly pass several configuration
parameters to ``spark-submit``::

    HAIL_HOME=$(pip3 show hail | grep Location | awk -F' ' '{print $2 "/hail"}')
    spark-submit \
      --jars $HAIL_HOME/hail-all-spark.jar \
      --conf spark.driver.extraClassPath=$HAIL_HOME/hail-all-spark.jar \
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
   version of Spark and the associated py4j version. The following example
   builds a Hail JAR for Cloudera's
   2.4.0 version of Spark::

    make install-on-cluster HAIL_COMPILE_NATIVES=1 SPARK_VERSION=2.4.0.cloudera PY4J_VERSION=0.10.7

 - On a Cloudera cluster, ``SPARK_HOME`` should be set as:
   ``SPARK_HOME=/opt/cloudera/parcels/SPARK2/lib/spark2``,

 - On Cloudera, you can create an interactive Python shell using ``pyspark``::

    HAIL_HOME=$(pip3 show hail | grep Location | awk -F' ' '{print $2 "/hail"}')
    spark-submit \
      --jars $HAIL_HOME/hail-all-spark.jar \
      --conf spark.driver.extraClassPath=$HAIL_HOME/hail-all-spark.jar \
      --conf spark.executor.extraClassPath=./hail-all-spark.jar \
      --conf spark.serializer=org.apache.spark.serializer.KryoSerializer \
      --conf spark.kryo.registrator=is.hail.kryo.HailKryoRegistrator \
      your-hail-python-script-here.py


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
