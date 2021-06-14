.. _CADD:

CADD
====

*  **Versions:** 1.4, 1.6
*  **Reference genome builds:** GRCh37, GRCh38
*  **Type:** :class:`hail.Table`

Schema (1.4, GRCh37)
~~~~~~~~~~~~~~~~~~~~

.. code-block:: text

    ----------------------------------------
    Global fields:
        'metadata': struct {
            name: str,
            version: str,
            reference_genome: str,
            n_rows: int64,
            n_partitions: int32
        }
    ----------------------------------------
    Row fields:
        'locus': locus<GRCh37>
        'alleles': array<str>
        'raw_score': float64
        'PHRED_score': float64
    ----------------------------------------
    Key: ['locus', 'alleles']
    ----------------------------------------
