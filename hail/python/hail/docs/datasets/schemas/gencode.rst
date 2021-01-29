.. _gencode:

gencode
=======

*  **Versions:** v19, v31
*  **Reference genome builds:** GRCh37, GRCh38
*  **Type:** :class:`hail.Table`

Schema (v19, GRCh37)
~~~~~~~~~~~~~~~~~~~~

.. code-block:: text

    ----------------------------------------
    Global fields:
        None
    ----------------------------------------
    Row fields:
        'interval': interval<locus<GRCh37>>
        'ID': str
        'gene_id': str
        'gene_name': str
        'gene_type': str
        'level': str
        'type': str
        'gene_score': str
        'gene_strand': str
        'gene_phase': str
    ----------------------------------------
    Key: ['interval']
    ----------------------------------------

