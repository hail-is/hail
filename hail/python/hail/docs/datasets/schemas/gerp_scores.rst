.. _gerp_scores:

gerp_scores
===========

*  **Versions:** hg19
*  **Reference genome builds:** GRCh37, GRCh38
*  **Type:** :class:`hail.Table`

Schema (hg19, GRCh37)
~~~~~~~~~~~~~~~~~~~~~

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
        'N': float64
        'S': float64
    ----------------------------------------
    Key: ['locus']
    ----------------------------------------

