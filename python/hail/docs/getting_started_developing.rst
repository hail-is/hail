.. _sec-getting_started_developing:

==========================
Getting Started Developing
==========================

You'll need:

- `Java 8 JDK <http://www.oracle.com/technetwork/java/javase/downloads/jdk8-downloads-2133151.html>`_.
- `Spark 2.2.0 <https://www.apache.org/dyn/closer.lua/spark/spark-2.2.0/spark-2.2.0-bin-hadoop2.7.tgz>`_.
- `Anaconda for Python 3 <https://www.continuum.io/downloads>`_.

-------------------
Building a Hail JAR
-------------------

The only additional tool necessary to build Hail from source is a C++ compiler. On a Debian-based OS like Ubuntu, a C++ compiler can be installed with apt-get::

    sudo apt-get install g++

On Mac OS X, a C++ compiler is provided by the Apple Xcode::

    xcode-select --install

The Hail source code is hosted `on GitHub <https://github.com/hail-is/hail>`_::

    git clone https://github.com/hail-is/hail.git
    cd hail

A Hail JAR can be built using Gradle. Note that every Hail JAR is specific to
one version of Spark::

    ./gradlew -Dspark.version=2.2.0 shadowJar

Finally, some environment variables must be set so that Hail can find Spark, Spark can find Hail, and Python can find Hail. Add these lines to your ``.bashrc`` or equivalent setting ``SPARK_HOME`` to the root directory of a Spark installation and ``HAIL_HOME`` to the root of the Hail repository::

    export SPARK_HOME=/path/to/spark
    export HAIL_HOME=/path/to/hail
    export PYTHONPATH="$PYTHONPATH:$HAIL_HOME/python:$SPARK_HOME/python:`echo $SPARK_HOME/python/lib/py4j*-src.zip`"
    export SPARK_CLASSPATH=$HAIL_HOME/build/libs/hail-all-spark.jar

Now you can import hail from a python interpreter::

    $ python
    Python 3.6.5 |Anaconda, Inc.| (default, Mar 29 2018, 13:14:23)
    [GCC 4.2.1 Compatible Clang 4.0.1 (tags/RELEASE_401/final)] on darwin
    Type "help", "copyright", "credits" or "license" for more information.

    >>> import hail as hl

    >>> hl.init() # doctest: +SKIP
    Using Spark's default log4j profile: org/apache/spark/log4j-defaults.properties
    Setting default log level to "WARN".
    To adjust logging level use sc.setLogLevel(newLevel). For SparkR, use setLogLevel(newLevel).
    Running on Apache Spark version 2.2.0
    SparkUI available at http://10.1.6.36:4041
    Welcome to
         __  __     <>__
        / /_/ /__  __/ /
       / __  / _ `/ / /
      /_/ /_/\_,_/_/_/   version devel-9f866ba
    NOTE: This is a beta version. Interfaces may change
      during the beta period. We also recommend pulling
      the latest changes weekly.

    >>>

-----------------
Building the Docs
-----------------

Hail uses `conda environments <https://conda.io/docs/using/envs.html>`_ to
manage the doc build process's python dependencies. First, create a conda
environment for hail:

.. code-block:: bash

    conda env create haildoc -f ./python/hail/dev-environment.yml

Activate the environment

.. code-block:: bash

    source activate haildoc

Now the shell prompt should include the name of the environment, in this case
"haildoc". Within the environment, run the ``makeDocs`` gradle task in the
environment:

.. code-block:: bash

    ./gradlew makeDocs

The generated docs are located at ``./build/www/hail/index.html``.

When you are finished developing hail, disable the environment

.. code-block:: bash

    source deactivate haildoc

The ``dev-environment.yml`` file may change without warning; therefore, after
pulling new changes from a remote repository, we always recommend updating the
conda environment

.. code-block:: bash

    conda env update haildoc -f ./python/hail/dev-environment.yml

-----------------
Running the tests
-----------------

Several Hail tests have additional dependencies:

 - `PLINK 1.9 <http://www.cog-genomics.org/plink2>`_

 - `QCTOOL 1.4 <http://www.well.ox.ac.uk/~gav/qctool>`_

 - `R 3.3.4 <http://www.r-project.org/>`_ with CRAN packages ``jsonlite``, ``SKAT`` and ``logistf``,
   as well as `pcrelate <https://www.rdocumentation.org/packages/GENESIS/versions/2.2.2/topics/pcrelate>`__
   from the `GENESIS <https://bioconductor.org/packages/release/bioc/html/GENESIS.html>`__ *Bioconductor* package.
   These can be installed within R using:
 
   .. code-block:: R

      install.packages(c("jsonlite", "SKAT", "logistf"))
      source("https://bioconductor.org/biocLite.R")
      biocLite("GENESIS")
      biocLite("SNPRelate")
      biocLite("GWASTools")

To execute all Hail tests, run:

.. code-block:: bash

    ./gradlew -Dspark.version=${SPARK_VERSION} -Dspark.home=${SPARK_HOME} test

