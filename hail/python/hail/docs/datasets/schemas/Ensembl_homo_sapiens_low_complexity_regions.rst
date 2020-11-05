.. _Ensembl_homo_sapiens_low_complexity_regions:

Ensembl_homo_sapiens_low_complexity_regions
===========================================

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
        'interval': interval<locus<GRCh37>>
    ----------------------------------------
    Key: ['interval']
    ----------------------------------------

