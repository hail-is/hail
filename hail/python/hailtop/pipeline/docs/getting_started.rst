.. _sec-getting_started:

===============
Getting Started
===============

Installation
------------

Pipeline is a Python module available inside the Hail Python package located
at `hailtop.pipeline`.


Installing Pipeline on Mac OS X or GNU/Linux with pip
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Create a `conda enviroment
<https://conda.io/docs/user-guide/concepts.html#conda-environments>`__ named
``hail`` and install the Hail python library in that environment. If ``conda activate`` doesn't work, `please read these instructions <https://conda.io/projects/conda/en/latest/user-guide/install/macos.html#install-macos-silent>`_

.. code-block:: sh

    conda create -n hail python'>=3.6,<3.8'
    conda activate hail
    pip install hail


To try Pipeline out, open iPython or a Jupyter notebook and run:

.. code-block:: python

    >>> import hailtop.pipeline as hp
    >>> p = hp.Pipeline()
    >>> t = p.new_task(name='hello')
    >>> t.command('echo "hello world"')
    >>> p.run()

You're now all set to run the :ref:`tutorial <sec-tutorial>`!
