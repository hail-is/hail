.. _sec-api:

==========
Python API
==========

This is the API documentation for ``Hail``, and provides detailed information
on the Python programming interface.

Use ``from hail import *`` to access this functionality.

.. autosummary::
    :nosignatures:
    :template: class.rst

    hail.Table
    hail.GroupedTable
    hail.MatrixTable
    hail.GroupedMatrixTable

.. rubric:: Modules

.. toctree::
    :maxdepth: 1

    genetics <genetics/index>
    expr <expr/index>
    methods <methods/index>
    utils <utils/index>
    linalg <linalg/index>
    stats <stats/index>

.. rubric:: Functions

.. autofunction:: hail.stop
.. autofunction:: hail.default_reference
