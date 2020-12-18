For Software Developers
-----------------------

Hail is an open-source project. We welcome contributions to the repository.

Requirements
~~~~~~~~~~~~

- `Java 8 JDK <https://adoptopenjdk.net/index.html>`_
  Note: it *must* be Java **8**. Hail does not support versions 9+ due to our
  dependency on Spark.

- The Python and non-pip installation requirements in `Getting Started <getting_started.html>`_

- If you are setting `HAIL_COMPILE_NATIVES=1`, then you need the LZ4 library
  header files. On Debian and Ubuntu machines run: `apt-get install liblz4-dev`.

Building Hail
~~~~~~~~~~~~~

The Hail source code is hosted `on GitHub <https://github.com/hail-is/hail>`_::

    git clone https://github.com/hail-is/hail.git
    cd hail/hail

By default, Hail uses pre-compiled native libraries that are compatible with
recent Mac OS X and Debian releases. If you're not using one of these OSes, set
the environment (or Make) variable `HAIL_COMPILE_NATIVES` to any value. This
variable tells GNU Make to build the native libraries from source.

Build and install a wheel file from source with local-mode ``pyspark``::

    make install HAIL_COMPILE_NATIVES=1

As above, but explicitly specifying the Scala and Spark versions::

    make install HAIL_COMPILE_NATIVES=1 SCALA_VERSION=2.11.12 SPARK_VERSION=2.4.5

Building the Docs and Website
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Install build dependencies listed in the `docs style guide <https://github.com/hail-is/hail/blob/main/hail/python/hail/docs/style-guide.txt>`_.

Build without testing the documentation examples::

    make hail-docs-no-test

Build while testing the documentation examples (significantly slower)::

    make hail-docs

Serve the built website on http://localhost:8000/ ::

    (cd build/www && python3 -m http.server)


Running the tests
~~~~~~~~~~~~~~~~~

Install development dependencies::

    make install-dev-deps

A couple Hail tests compare to PLINK 1.9 (not PLINK 2.0 [ignore the confusing
URL]):

 - `PLINK 1.9 <https://www.cog-genomics.org/plink2>`_

Execute every Hail test using at most 8 parallel threads::

    make -j8 test

Contributing
~~~~~~~~~~~~

Chat with the dev team on our `Zulip chatroom <https://hail.zulipchat.com>`_ or
`development forum <https://dev.hail.is>`_ if you have an idea for a contribution.
We can help you determine if your project is a good candidate for merging.

Keep in mind the following principles when submitting a pull request:

- A PR should focus on a single feature. Multiple features should be split into multiple PRs.
- Before submitting your PR, you should rebase onto the latest main.
- PRs must pass all tests before being merged. See the section above on `Running the tests`_ locally.
- PRs require a review before being merged. We will assign someone from our dev team to review your PR.
- When you make a PR, include a short message that describes the purpose of the
  PR and any necessary context for the changes you are making.
- For user facing changes (new functions, etc), include "CHANGELOG" in the commit message or PR title.
  This helps identify what should be included in the change log when a new version is released.
