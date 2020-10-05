.. _gnomad_lof_metrics:

gnomad_lof_metrics
==================

*  **Versions:** 2.1.1
*  **Reference genome builds:** None
*  **Type:** :class:`.Table`

Schema (2.1.1, None)
~~~~~~~~~~~~~~~~~~~~

.. code-block:: text

    ----------------------------------------
    Global fields:
        None
    ----------------------------------------
    Row fields:
        'gene': str
        'transcript': str
        'obs_mis': int32
        'exp_mis': float64
        'oe_mis': float64
        'mu_mis': float64
        'possible_mis': int32
        'obs_mis_pphen': int32
        'exp_mis_pphen': float64
        'oe_mis_pphen': float64
        'possible_mis_pphen': int32
        'obs_syn': int32
        'exp_syn': float64
        'oe_syn': float64
        'mu_syn': float64
        'possible_syn': int32
        'obs_lof': int32
        'mu_lof': float64
        'possible_lof': int32
        'exp_lof': float64
        'pLI': float64
        'pNull': float64
        'pRec': float64
        'oe_lof': float64
        'oe_syn_lower': float64
        'oe_syn_upper': float64
        'oe_mis_lower': float64
        'oe_mis_upper': float64
        'oe_lof_lower': float64
        'oe_lof_upper': float64
        'constraint_flag': str
        'syn_z': float64
        'mis_z': float64
        'lof_z': float64
        'oe_lof_upper_rank': int32
        'oe_lof_upper_bin': int32
        'oe_lof_upper_bin_6': int32
        'n_sites': int32
        'classic_caf': float64
        'max_af': float64
        'no_lofs': int32
        'obs_het_lof': int32
        'obs_hom_lof': int32
        'defined': int32
        'p': float64
        'exp_hom_lof': float64
        'classic_caf_afr': float64
        'classic_caf_amr': float64
        'classic_caf_asj': float64
        'classic_caf_eas': float64
        'classic_caf_fin': float64
        'classic_caf_nfe': float64
        'classic_caf_oth': float64
        'classic_caf_sas': float64
        'p_afr': float64
        'p_amr': float64
        'p_asj': float64
        'p_eas': float64
        'p_fin': float64
        'p_nfe': float64
        'p_oth': float64
        'p_sas': float64
        'transcript_type': str
        'gene_id': str
        'transcript_level': int32
        'cds_length': int32
        'num_coding_exons': int32
        'gene_type': str
        'gene_length': int32
        'exac_pLI': float64
        'exac_obs_lof': int32
        'exac_exp_lof': float64
        'exac_oe_lof': float64
        'brain_expression': str
        'chromosome': str
        'start_position': int32
        'end_position': int32
    ----------------------------------------
    Key: ['gene']
    ----------------------------------------

