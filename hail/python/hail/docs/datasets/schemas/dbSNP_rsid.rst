.. _dbSNP_rsid:

dbSNP_rsid
==========

*  **Versions:** 154
*  **Reference genome builds:** GRCh37, GRCh38
*  **Type:** :class:`hail.Table`

Schema (154, GRCh37)
~~~~~~~~~~~~~~~~~~~~

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
        'rsid': str
    ----------------------------------------
    Key: ['locus', 'alleles']
    ----------------------------------------
