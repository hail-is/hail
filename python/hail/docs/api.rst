.. _sec-api:

==========
Python API
==========

This is the API documentation for ``Hail``, and provides detailed information
on the Python programming interface.

Use ``import hail as hl`` to access this functionality.

.. autosummary::
    :nosignatures:
    :toctree: ./
    :template: class.rst

    hail.Table
    hail.GroupedTable
    hail.MatrixTable
    hail.GroupedMatrixTable

.. rubric:: Modules

.. toctree::
    :maxdepth: 1

    expressions <expressions>
    types <types>
    functions <functions/index>
    aggregators <aggregators>
    methods <methods/index>
    utils <utils/index>
    linalg <linalg/index>
    stats <stats/index>
    genetics <genetics/index>
    plot <plot>
    experimental <experimental>

.. rubric:: Module functions

.. autofunction:: hail.init
.. autofunction:: hail.stop
.. autofunction:: hail.spark_context
.. autofunction:: hail.default_reference
.. autofunction:: hail.get_reference
