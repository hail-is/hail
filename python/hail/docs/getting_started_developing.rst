.. _sec-getting_started_developing:

==========================
Getting Started Developing
==========================

You'll need:

- The `Java 8 JDK <http://www.oracle.com/technetwork/java/javase/downloads/index.html>`_.
- `Spark 2.0.2 <http://spark.apache.org/downloads.html>`_
- `Anaconda <https://www.continuum.io/downloads>`_.

-------------------
Building a Hail JAR
-------------------

See :ref:`building-hail-from-source`.

-----------------
Building the Docs
-----------------

Hail uses conda environments to manage python dependencies. First, create a
conda environment for hail:

.. code-block:: text

    conda env create hail -f ./python/hail/environment.yml

And run the ``createDocs`` gradle task in the environment:

.. code-block:: text

    (source activate hail && gradle createDocs) ; source deactivate hail

The generated docs are located at ``./build/www/hail/index.html``.

The ``environment.yml`` file may change without warning; therefore, after
creating the environment, we recommend building the docs as follows to ensure
the environment is always up to date:

.. code-block:: text

    (conda env update hail -f ./python/hail/environment.yml &&
     source activate hail &&
     gradle createDocs) ; source deactivate hail

-----------------
Running the tests
-----------------

Several Hail tests have additional dependencies:

 - `PLINK 1.9 <http://www.cog-genomics.org/plink2>`_

 - `QCTOOL 1.4 <http://www.well.ox.ac.uk/~gav/qctool>`_

 - `R 3.3.1 <http://www.r-project.org/>`_ with packages ``jsonlite`` and ``logistf``, which depends on ``mice`` and ``Rcpp``.

Other recent versions of QCTOOL and R should suffice, but PLINK 1.7 will not.

To execute all Hail tests, run:

.. code-block:: text

    $ ./gradlew -Dspark.home=${SPARK_HOME} test
