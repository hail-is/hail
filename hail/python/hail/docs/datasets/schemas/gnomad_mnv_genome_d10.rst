.. _gnomad_mnv_genome_d10:

gnomad_mnv_genome_d10
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
        'locus': locus<GRCh37>
        'refs': str
        'alts': str
        'distance': int32
        'snp1': str
        'snp2': str
        'ac1': int32
        'ac2': int32
        'ac_mnv': int32
        'ac1_adj': int32
        'ac2_adj': int32
        'ac_mnv_adj': int32
    ----------------------------------------
    Key: ['locus', 'refs', 'alts']
    ----------------------------------------

