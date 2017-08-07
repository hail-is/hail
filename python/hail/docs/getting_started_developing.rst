.. _sec-getting_started_developing:

==========================
Getting Started Developing
==========================

You'll need:

- `Java 8 JDK <http://www.oracle.com/technetwork/java/javase/downloads/index.html>`_
- `Spark 2.0.2 <http://spark.apache.org/downloads.html>`_
- `Anaconda <https://www.continuum.io/downloads>`_

-------------------
Building a Hail JAR
-------------------

See :ref:`building-hail-from-source`.

-----------------
Building the Docs
-----------------

Hail uses `conda environments <https://conda.io/docs/using/envs.html>`_ to
manage python dependencies. First, create a conda environment for hail:

.. code-block:: bash

    conda env create hail -f ./python/hail/environment.yml

Activate the environment

.. code-block:: bash

    source activate hail

Now the shell prompt should include the name of the environment, in this case
"hail". Within the environment, run the ``createDocs`` gradle task in the
environment:

.. code-block:: bash

    ./gradlew createDocs

The generated docs are located at ``./build/www/hail/index.html``.

When you are finished developing hail, disable the environment

.. code-block:: bash

    source deactivate hail

The ``environment.yml`` file may change without warning; therefore, after
pulling new changes from a remote repository, we always recommend updating the
conda environment

.. code-block:: bash

    conda env update hail -f ./python/hail/environment.yml

-----------------
Running the tests
-----------------

Several Hail tests have additional dependencies:

 - `PLINK 1.9 <http://www.cog-genomics.org/plink2>`_

 - `QCTOOL 1.4 <http://www.well.ox.ac.uk/~gav/qctool>`_

 - `R 3.3.1 <http://www.r-project.org/>`_ with packages ``jsonlite`` and ``logistf``, which depends on ``mice`` and ``Rcpp``.

Other recent versions of QCTOOL and R should suffice, but PLINK 1.7 will not.

To execute all Hail tests, run:

.. code-block:: bash

    ./gradlew -Dspark.home=${SPARK_HOME} test

