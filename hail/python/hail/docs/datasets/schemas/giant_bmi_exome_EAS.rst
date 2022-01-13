.. _giant_bmi_exome_EAS:

giant_bmi_exome_EAS
===================

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
        'eas_maf': dict<str, float64>
        'exac_eas_maf': dict<str, float64>
        'beta': float64
        'se': float64
        'pvalue': float64
    ----------------------------------------
    Key: ['locus', 'alleles']
    ----------------------------------------
