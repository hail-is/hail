.. _gnomad_exome_coverage:

gnomad_exome_coverage
=====================

*  **Versions:** 2.1
*  **Reference genome builds:** GRCh37
*  **Type:** :class:`hail.Table`

Schema (2.1, GRCh37)
~~~~~~~~~~~~~~~~~~~~

.. code-block:: text

    ----------------------------------------
    Global fields:
        None
    ----------------------------------------
    Row fields:
        'row_id': int64
        'locus': locus<GRCh37>
        'mean': float64
        'median': int32
        'over_1': float64
        'over_5': float64
        'over_10': float64
        'over_15': float64
        'over_20': float64
        'over_25': float64
        'over_30': float64
        'over_50': float64
        'over_100': float64
    ----------------------------------------
    Key: ['locus']
    ----------------------------------------

