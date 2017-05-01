.. _sec-api:

==========
Python API
==========

This is the API documentation for ``Hail``, and provides detailed information
on the Python programming interface. See the :ref:`tutorial.ipynb` for an
introduction to using this API to analyze genetic data.



.. rubric:: Classes

.. autosummary::
    :nosignatures:
    :toctree: ./
    :template: class.rst

    hail.HailContext
    hail.VariantDataset
    hail.KeyTable
    hail.TextTableConfig
    hail.KinshipMatrix


.. rubric:: Modules

.. toctree::
    :maxdepth: 1

    representation <representation/index>
    expr <expr/index>
    utils <utils/index>
