.. _sec-query-api:

=====================
Hail Query Python API
=====================

This is the API documentation for ``Hail Query``, and provides detailed information
on the Python programming interface.

Use ``import hail as hl`` to access this functionality.

Classes
~~~~~~~

.. autosummary::
    :nosignatures:
    :toctree: ./
    :template: class.rst

    hail.Table
    hail.GroupedTable
    hail.MatrixTable
    hail.GroupedMatrixTable

Modules
~~~~~~~

.. toctree::
    :maxdepth: 1

    expressions <expressions>
    types <types>
    functions <functions/index>
    aggregators <aggregators>
    scans <scans>
    methods <methods/index>
    nd <nd/index>
    utils <utils/index>
    linalg <linalg/index>
    stats <stats/index>
    genetics <genetics/index>
    plot <plot>
    ggplot <ggplot/index>
    vds <vds/index>
    experimental <experimental/index>

Top-Level Functions
~~~~~~~~~~~~~~~~~~~

.. autofunction:: hail.init
.. autofunction:: hail.asc
.. autofunction:: hail.desc
.. autofunction:: hail.stop
.. autofunction:: hail.spark_context
.. autofunction:: hail.tmp_dir
.. autofunction:: hail.default_reference
.. autofunction:: hail.get_reference
.. autofunction:: hail.set_global_seed
.. autofunction:: hail.reset_global_randomness
.. autofunction:: hail.citation
.. autofunction:: hail.version
