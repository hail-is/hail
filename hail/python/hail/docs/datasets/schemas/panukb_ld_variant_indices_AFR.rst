.. _panukb_ld_variant_indices_AFR:

panukb_ld_variant_indices_AFR
=============================

*  **Versions:** 0.2
*  **Reference genome builds:** GRCh37
*  **Type:** :class:`hail.Table`

Schema (0.2, GRCh37)
~~~~~~~~~~~~~~~~~~~~

.. code-block:: text

    ----------------------------------------
    Global fields:
        'n_samples': int32
        'pop': str
    ----------------------------------------
    Row fields:
        'locus': locus<GRCh37>
        'alleles': array<str>
        'idx': int64
    ----------------------------------------
    Key: ['locus', 'alleles']
    ----------------------------------------
