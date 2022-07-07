.. _gencode:

gencode
=======

*  **Versions:** v19, v31, v35
*  **Reference genome builds:** GRCh37, GRCh38
*  **Type:** :class:`hail.Table`

Schema (v35, GRCh38)
~~~~~~~~~~~~~~~~~~~~

.. code-block:: text

    ----------------------------------------
    Global fields:
        None
    ----------------------------------------
    Row fields:
        'interval': interval<locus<GRCh38>>
        'source': str
        'feature': str
        'score': float64
        'strand': str
        'frame': int32
        'tag': str
        'level': int32
        'gene_id': str
        'gene_type': str
        'ccdsid': str
        'exon_id': str
        'exon_number': int32
        'havana_gene': str
        'transcript_type': str
        'protein_id': str
        'gene_name': str
        'transcript_name': str
        'transcript_id': str
        'transcript_support_level': str
        'hgnc_id': str
        'ont': str
        'havana_transcript': str
    ----------------------------------------
    Key: ['interval']
    ----------------------------------------
