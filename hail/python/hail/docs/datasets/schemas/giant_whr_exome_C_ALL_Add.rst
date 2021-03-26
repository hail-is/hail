.. _giant_whr_exome_C_ALL_Add:

giant_whr_exome_C_ALL_Add
=========================

*  **Versions:** 2018
*  **Reference genome builds:** GRCh37
*  **Type:** :class:`hail.Table`

Schema (2018, GRCh37)
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: text

    ----------------------------------------
    Global fields:
        'metadata': struct {
            name: str,
            version: str,
            reference_genome: str,
            n_rows: int32,
            n_partitions: int32
        }
    ----------------------------------------
    Row fields:
        'locus': locus<GRCh37>
        'alleles': array<str>
        'snp_name': str
        'gmaf': dict<str, float64>
        'exac_maf': dict<str, float64>
        'beta': float64
        'se': float64
        'pvalue': float64
        'sample_size': int32
    ----------------------------------------
    Key: ['locus', 'alleles']
    ----------------------------------------
