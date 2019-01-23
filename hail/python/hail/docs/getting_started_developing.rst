For Software Developers
-----------------------

Hail is an open-source project. We welcome contributions to the repository. If you're interested
in contributing to Hail, you will need to build your own Hail JAR and set up the testing environment.

Requirements
~~~~~~~~~~~~

You'll need:

- `Java 8 JDK <http://www.oracle.com/technetwork/java/javase/downloads/jdk8-downloads-2133151.html>`_
- Python 3.6 or later, we recommend `Anaconda's Python 3 <https://www.anaconda.com/download/>`_

Building a Hail JAR
~~~~~~~~~~~~~~~~~~~

To build Hail from source, you will need a C++ compiler and lz4. From the root
of the Hail repository, install a C++ compiler and lz4, as well as other
essential hail environment things::

    ./env-setup.sh

Build a Hail jar compatible with Spark 2.2.0::

    SPARK_VERSION=2.2.0 make jar


Conda Environments
~~~~~~~~~~~~~~~~~~

Hail uses `conda environments <https://conda.io/docs/using/envs.html>`_ to
manage python dependencies. You should not need to directly manipulate them. The
Makefile handles creating, updating, activating, and deactivating them.


Installing Hail Locally
~~~~~~~~~~~~~~~~~~~~~~~

Install the currently checked out version of hail (you may want to do this from
within a conda environment or virtualenv)::

    make pip-install

Spark Configuration
~~~~~~~~~~~~~~~~~~~

You may find it helpful to increase the memory available to Spark::

    export PYSPARK_SUBMIT_ARGS="--driver-memory 8G pyspark-shell"


Building the Docs
~~~~~~~~~~~~~~~~~

.. code-block:: bash

    make docs

The generated docs are located at ``./build/www/docs/0.2/index.html``.


Running the tests
~~~~~~~~~~~~~~~~~

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

    make test

Contributing
~~~~~~~~~~~~

Chat with the dev team on our `Zulip chatroom <https://hail.zulipchat.com>`_ if
you have an idea for a contribution. We can help you determine if your
project is a good candidate for merging.

Keep in mind the following principles when submitting a pull request:

- A PR should focus on a single feature. Multiple features should be split into multiple PRs.
- Before submitting your PR, you should rebase onto the latest master.
- PRs must pass all tests before being merged. See the section above on `Running the tests`_ locally.
- PRs require a review before being merged. We will assign someone from our dev team to review your PR.
- Code in PRs should be formatted according to the style in ``code_style.xml``.
  This file can be loaded into Intellij to automatically format your code.
- When you make a PR, include a short message that describes the purpose of the
  PR and any necessary context for the changes you are making.
