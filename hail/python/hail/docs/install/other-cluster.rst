===============================
Install Hail on a Spark Cluster
===============================

If you are using Google Dataproc, please see `these simpler instructions <dataproc.rst>`__. If you
are using Azure HDInsight please see `these simpler instructions <azure.rst>`__.

Hail should work with any Spark 3.1.1 cluster built with Scala 2.12.

Hail needs to be built from source on the leader node. Building Hail from source
requires:

- Java 8 or 11 JDK.
- Python 3.7 or later.
- A recent C and a C++ compiler, GCC 5.0, LLVM 3.4, or later versions of either
  suffice.
- The LZ4 library.
- BLAS and LAPACK.

On a Debian-like system, the following should suffice:

.. code-block:: sh

   apt-get update
   apt-get install \
       openjdk-8-jdk-headless \
       g++ \
       python3 python3-pip \
       libopenblas-dev liblapack-dev \
       liblz4-dev


The next block of commands downloads, builds, and installs Hail from source.

.. code-block:: sh

    git clone https://github.com/hail-is/hail.git
    cd hail/hail
    make install-on-cluster HAIL_COMPILE_NATIVES=1 SCALA_VERSION=2.12.15 SPARK_VERSION=3.3.0

If you forget to install any of the requirements before running `make install-on-cluster`, it's possible
to get into a bad state where `make` insists you don't have a requirement that you have in fact installed.
Try doing `make clean` and then a fresh invocation of the `make install-on-cluster` line if this happens.

On every worker node of the cluster, you must install a BLAS and LAPACK library
such as the Intel MKL or OpenBLAS. On a Debian-like system you might try the
following on every worker node.

.. code-block:: sh

   apt-get install libopenblas liblapack3

Hail is now installed! You can use ``ipython``, ``python``, and ``jupyter
notebook`` without any further configuration. We recommend against using the
``pyspark`` command.

Let's take Hail for a spin! Create a file called "hail-script.py" and place the
following analysis of a randomly generated dataset with five-hundred samples and
half-a-million variants.

.. code-block:: python3

    import hail as hl
    mt = hl.balding_nichols_model(n_populations=3,
                                  n_samples=500,
                                  n_variants=500_000,
                                  n_partitions=32)
    mt = mt.annotate_cols(drinks_coffee = hl.rand_bool(0.33))
    gwas = hl.linear_regression_rows(y=mt.drinks_coffee,
                                     x=mt.GT.n_alt_alleles(),
                                     covariates=[1.0])
    gwas.order_by(gwas.p_value).show(25)

Run the script and wait for the results. You should not have to wait more than a
minute.

.. code-block:: sh

   python3 hail-script.py

Slightly more configuration is necessary to ``spark-submit`` a Hail script:

.. code-block:: sh

    HAIL_HOME=$(pip3 show hail | grep Location | awk -F' ' '{print $2 "/hail"}')
    spark-submit \
      --jars $HAIL_HOME/hail-all-spark.jar \
      --conf spark.driver.extraClassPath=$HAIL_HOME/hail-all-spark.jar \
      --conf spark.executor.extraClassPath=./hail-all-spark.jar \
      --conf spark.serializer=org.apache.spark.serializer.KryoSerializer \
      --conf spark.kryo.registrator=is.hail.kryo.HailKryoRegistrator \
      hail-script.py

Next Steps
""""""""""

- Get the `Hail cheatsheets <../cheatsheets.rst>`__
- Follow the Hail `GWAS Tutorial <../tutorials/01-genome-wide-association-study.rst>`__
