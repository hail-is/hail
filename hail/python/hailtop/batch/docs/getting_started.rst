.. _sec-getting_started:

===============
Getting Started
===============

Installation
------------

Batch is a Python module available inside the Hail Python package located at `hailtop.batch`. The
Batch Service additionally depends on the Google Cloud SDK.


Installing Batch on Mac OS X or GNU/Linux with pip
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Create a `conda enviroment
<https://conda.io/docs/user-guide/concepts.html#conda-environments>`__ named
``hail`` and install the Hail python library in that environment. If ``conda activate`` doesn't work, `please read these instructions <https://conda.io/projects/conda/en/latest/user-guide/install/macos.html#install-macos-silent>`_

.. code-block:: sh

    conda create -n hail python'>=3.8'
    conda activate hail
    pip install hail

Installing the Google Cloud SDK
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If you plan to use the Batch Service (as opposed to the local-only mode), then you must additionally
`install the Google Cloud SDK <https://cloud.google.com/sdk/docs/install>`__.

Try it out!
~~~~~~~~~~~

To try `batch` out, open iPython or a Jupyter notebook and run:

.. code-block:: python

    >>> import hailtop.batch as hb
    >>> b = hb.Batch()
    >>> j = b.new_job(name='hello')
    >>> j.command('echo "hello world"')
    >>> b.run()

You're now all set to run the :ref:`tutorial <sec-tutorial>`!
