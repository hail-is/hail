.. _GTEx_eQTL_associations:

GTEx_eQTL_associations
======================

*  **Versions:** v7
*  **Reference genome builds:** GRCh37
*  **Type:** :class:`MatrixTable`

Schema (v7, GRCh37)
~~~~~~~~~~~~~~~~~~~

.. code-block:: text

    ----------------------------------------
    Global fields:
        'metadata': struct {
            name: str, 
            version: str, 
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
        'locus': locus<GRCh37> 
        'alleles': array<str> 
        'gene_id': str 
        'tss_distance': int32 
        'maf': float64 
        'gene_interval': interval<locus<GRCh37>> 
        'gene_name': str 
        'strand': str 
        'rsid': str 
    ----------------------------------------
    Entry fields:
        'ma_samples': int32 
        'ma_count': int32 
        'pval_nominal': float64 
        'slope': float64 
        'slope_se': float64 
        'pval_nominal_threshold': float64 
        'min_pval_nominal': float64 
        'pval_beta': float64 
        'is_significant': bool 
        'is_most_significant_variant_per_gene': bool 
        'eGenes': struct {
            num_var: int32, 
            beta_shape1: float64, 
            beta_shape2: float64, 
            true_df: float64, 
            pval_true_df: float64, 
            num_alt_per_site: int32, 
            ref_factor: int32, 
            pval_perm: float64, 
            pval_beta: float64, 
            qval: float64, 
            log2_aFC: float64, 
            log2_aFC_lower: float64, 
            log2_aFC_upper: str
        } 
    ----------------------------------------
    Column key: ['tissue']
    Row key: ['locus', 'alleles', 'gene_id']
    ----------------------------------------
    
