.. _gerp_elements:

gerp_elements
=============

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
        'interval': interval<locus<GRCh37>>
        'S': float64
        'p_value': float64
    ----------------------------------------
    Key: ['interval']
    ----------------------------------------

