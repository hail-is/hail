.. _Ensembl_homo_sapiens_reference_genome:

Ensembl_homo_sapiens_reference_genome
=====================================

*  **Versions:** release_95
*  **Reference genome builds:** GRCh37, GRCh38
*  **Type:** :class:`hail.Table`

Schema (release_95, GRCh37)
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: text

    ----------------------------------------
    Global fields:
        'metadata': struct {
            name: str,
            version: str,
            reference_genome: str,
            n_partitions: int32
        }
    ----------------------------------------
    Row fields:
        'locus': locus<GRCh37>
        'reference_allele': str
    ----------------------------------------
    Key: ['locus']
    ----------------------------------------

