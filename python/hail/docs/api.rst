.. _sec-api:

==========
Python API
==========

This is the API documentation for ``Hail``, and provides detailed information
on the Python programming interface.


``api1`` is the old 0.1-style interface.

.. toctree::
    :maxdepth: 2

.. autosummary::
    :nosignatures:
    :toctree: ./api1/
    :template: class.rst

    hail.api1.HailContext
    hail.api1.KeyTable
    hail.api1.VariantDataset

``api2`` is the shiny new interface.

.. toctree::
    :maxdepth: 2

.. autosummary::
    :nosignatures:
    :toctree: ./api2/
    :template: class.rst

    hail.api2.context.HailContext
    hail.api2.table.Table
    hail.api2.table.GroupedTable
    hail.api2.matrixtable.MatrixTable
    hail.api2.matrixtable.GroupedMatrixTable

.. rubric:: Modules

.. toctree::
    :maxdepth: 1

    genetics <genetics/index>
    expr <expr/index>
    methods <methods/index>
    utils <utils/index>
