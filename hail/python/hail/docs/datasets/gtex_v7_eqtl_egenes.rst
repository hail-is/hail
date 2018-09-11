.. _gtex_v7_eqtl_egenes:

gtex_v7_eqtl_egenes
===================

.. code-block:: text

    ----------------------------------------
    Global fields:
    None
    ----------------------------------------
    Column fields:
    'tissue': str 
    ----------------------------------------
    Row fields:
    'locus': locus<GRCh37> 
    'alleles': array<str> 
    'gene_id': str 
    'rsid': str 
    'gene_interval': interval<locus<GRCh37>> 
    'gene_name': str 
    'strand': str 
    'tss_distance': int32 
    'maf': float64 
    ----------------------------------------
    Entry fields:
    'num_var': int32 
    'beta_shape1': float64 
    'beta_shape2': float64 
    'true_df': float64 
    'pval_true_df': float64 
    'num_alt_per_site': int32 
    'minor_allele_samples': int32 
    'minor_allele_count': int32 
    'ref_factor': int32 
    'pval_nominal': float64 
    'slope': float64 
    'slope_se': float64 
    'pval_perm': float64 
    'pval_beta': float64 
    'qval': float64 
    'pval_nominal_threshold': float64 
    'log2_aFC': float64 
    'log2_aFC_lower': float64 
    'log2_aFC_upper': str 
    ----------------------------------------
    Column key: ['tissue']
    Row key: ['locus', 'alleles', 'gene_id']
    ----------------------------------------
