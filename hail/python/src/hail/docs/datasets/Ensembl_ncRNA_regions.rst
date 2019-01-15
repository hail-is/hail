.. _Ensembl_ncRNA_regions:

Ensembl_ncRNA_regions
=====================

*  **Versions:** release_93
*  **Reference genome builds:** GRCh37, GRCh38
*  **Type:** :class:`Table`

Schema (release_93, GRCh37)
~~~~~~~~~~~~~~~~~~~~~~~~~~~

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
        'interval': interval<locus<GRCh37>> 
        'strand': str 
        'seqtype': str 
        'status': str 
        'transcript_id': str 
        'transcript_biotype': str 
        'gene_id': str 
        'gene_biotype': str 
    ----------------------------------------
    Key: ['interval']
    ----------------------------------------
    
