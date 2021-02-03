.. _gnomad_ld_scores_fin:

gnomad_ld_scores_fin
====================

*  **Versions:** 2.1.1
*  **Reference genome builds:** GRCh37
*  **Type:** :class:`hail.Table`

Schema (2.1.1, GRCh37)
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: text

    ----------------------------------------
    Global fields:
        None
    ----------------------------------------
    Row fields:
        'locus': locus<GRCh37>
        'alleles': array<str>
        'pop_freq': struct {
            AC: int32,
            AF: float64,
            AN: int32,
            homozygote_count: int32
        }
        'idx': int64
        'new_idx': int64
        'ld_score': float64
    ----------------------------------------
    Key: ['locus', 'alleles']
    ----------------------------------------

