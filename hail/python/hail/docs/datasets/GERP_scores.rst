.. _GERP_scores:

GERP_scores
===========

*  **Versions:** GERP++
*  **Reference genome builds:** GRCh37, GRCh38
*  **Type:** :class:`Table`

Schema (GERP++, GRCh37)
~~~~~~~~~~~~~~~~~~~~~~~

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
        'N': float64 
        'S': float64 
    ----------------------------------------
    Key: ['locus']
    ----------------------------------------
    
