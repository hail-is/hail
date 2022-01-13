.. _GTEx_eQTL_all_snp_gene_associations:

GTEx_eQTL_all_snp_gene_associations
===================================

*  **Versions:** v8
*  **Reference genome builds:** GRCh38
*  **Type:** :class:`hail.MatrixTable`

Schema (v8, GRCh38)
~~~~~~~~~~~~~~~~~~~

.. code-block:: text

    ----------------------------------------
    Global fields:
        'metadata': struct {
            name: str,
            reference_genome: str,
            n_rows: int32,
            n_cols: int32,
            n_partitions: int32
        }
    ----------------------------------------
    Column fields:
        'tissue': str
    ----------------------------------------
    Row fields:
        'locus': locus<GRCh38>
        'alleles': array<str>
        'gene_id': str
        'tss_distance': int32
    ----------------------------------------
    Entry fields:
        'ma_samples': int32
        'ma_count': int32
        'maf': float64
        'pval_nominal': float64
        'slope': float64
        'slope_se': float64
    ----------------------------------------
    Column key: ['tissue']
    Row key: ['locus', 'alleles']
    ----------------------------------------
