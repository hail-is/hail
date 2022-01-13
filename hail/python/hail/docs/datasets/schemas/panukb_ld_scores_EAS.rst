.. _panukb_ld_scores_EAS:

panukb_ld_scores_EAS
====================

*  **Versions:** 0.2
*  **Reference genome builds:** GRCh37
*  **Type:** :class:`hail.Table`

Schema (0.2, GRCh37)
~~~~~~~~~~~~~~~~~~~~

.. code-block:: text

    ----------------------------------------
    Global fields:
        None
    ----------------------------------------
    Row fields:
        'locus': locus<GRCh37>
        'alleles': array<str>
        'rsid': str
        'varid': str
        'AF': float64
        'ld_score': float64
    ----------------------------------------
    Key: ['locus', 'alleles']
    ----------------------------------------
