.. _Ensembl_human_reference_genome:

Ensembl_human_reference_genome
==============================

*  **Versions:** release_93
*  **Reference genome builds:** GRCh37, GRCh38
*  **Type:** Table

Schema (release_93, GRCh37)
~~~~~~~~~~~~~~~~~~~~~~~~~~~

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
        'reference_allele': str 
    ----------------------------------------
    Key: ['locus']
    ----------------------------------------
    
