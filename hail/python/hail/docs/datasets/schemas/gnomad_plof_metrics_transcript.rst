.. _gnomad_plof_metrics_transcript:

gnomad_plof_metrics_transcript
==============================

*  **Versions:** 2.1.1
*  **Reference genome builds:** GRCh37
*  **Type:** :class:`hail.Table`

Schema (2.1.1, GRCh37)
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: text

    ----------------------------------------
    Global fields:
        'rf': struct {
            variants_by_type: dict<str, int32>,
            feature_medians: dict<str, struct {
                variant_type: str,
                n_alt_alleles: int32,
                qd: float64,
                pab_max: float64,
                info_MQRankSum: float64,
                info_SOR: float64,
                info_InbreedingCoeff: float64,
                info_ReadPosRankSum: float64,
                info_FS: float64,
                info_QD: float64,
                info_MQ: float64,
                info_DP: int32
            }>,
            test_intervals: array<interval<locus<GRCh37>>>,
            test_results: array<struct {
                rf_prediction: str,
                rf_label: str,
                n: int32
            }>,
            features_importance: dict<str, float64>,
            features: array<str>,
            vqsr_training: bool,
            no_transmitted_singletons: bool,
            adj: bool,
            rf_hash: str,
            rf_snv_cutoff: struct {
                bin: int32,
                min_score: float64
            },
            rf_indel_cutoff: struct {
                bin: int32,
                min_score: float64
            }
        }
        'freq_meta': array<dict<str, str>>
        'freq_index_dict': dict<str, int32>
        'popmax_index_dict': dict<str, int32>
        'age_index_dict': dict<str, int32>
        'faf_index_dict': dict<str, int32>
        'syn_sd': float64
        'mis_sd': float64
        'lof_sd': float64
    ----------------------------------------
    Row fields:
        'gene': str
        'transcript': str
        'canonical': bool
        'obs_mis': int64
        'exp_mis': float64
        'oe_mis': float64
        'mu_mis': float64
        'possible_mis': int64
        'exp_mis_global': array<float64>
        'obs_mis_global': array<int64>
        'exp_mis_afr': array<float64>
        'obs_mis_afr': array<int64>
        'exp_mis_amr': array<float64>
        'obs_mis_amr': array<int64>
        'exp_mis_eas': array<float64>
        'obs_mis_eas': array<int64>
        'exp_mis_nfe': array<float64>
        'obs_mis_nfe': array<int64>
        'exp_mis_sas': array<float64>
        'obs_mis_sas': array<int64>
        'obs_mis_pphen': int64
        'exp_mis_pphen': float64
        'oe_mis_pphen': float64
        'possible_mis_pphen': int64
        'obs_syn': int64
        'exp_syn': float64
        'oe_syn': float64
        'mu_syn': float64
        'possible_syn': int64
        'exp_syn_global': array<float64>
        'obs_syn_global': array<int64>
        'exp_syn_afr': array<float64>
        'obs_syn_afr': array<int64>
        'exp_syn_amr': array<float64>
        'obs_syn_amr': array<int64>
        'exp_syn_eas': array<float64>
        'obs_syn_eas': array<int64>
        'exp_syn_nfe': array<float64>
        'obs_syn_nfe': array<int64>
        'exp_syn_sas': array<float64>
        'obs_syn_sas': array<int64>
        'obs_lof': int64
        'mu_lof': float64
        'possible_lof': int64
        'exp_lof': float64
        'exp_lof_global': array<float64>
        'obs_lof_global': array<int64>
        'exp_lof_afr': array<float64>
        'obs_lof_afr': array<int64>
        'exp_lof_amr': array<float64>
        'obs_lof_amr': array<int64>
        'exp_lof_eas': array<float64>
        'obs_lof_eas': array<int64>
        'exp_lof_nfe': array<float64>
        'obs_lof_nfe': array<int64>
        'exp_lof_sas': array<float64>
        'obs_lof_sas': array<int64>
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
        'constraint_flag': set<str>
        'syn_z': float64
        'mis_z': float64
        'lof_z': float64
        'oe_lof_upper_rank': int64
        'oe_lof_upper_bin': int32
        'oe_lof_upper_bin_6': int32
        'n_sites': int64
        'n_sites_array': array<int64>
        'classic_caf': float64
        'max_af': float64
        'classic_caf_array': array<float64>
        'no_lofs': int64
        'obs_het_lof': int64
        'obs_hom_lof': int64
        'defined': int64
        'pop_no_lofs': dict<str, int64>
        'pop_obs_het_lof': dict<str, int64>
        'pop_obs_hom_lof': dict<str, int64>
        'pop_defined': dict<str, int64>
        'p': float64
        'pop_p': dict<str, float64>
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
        'transcript_level': str
        'cds_length': int64
        'num_coding_exons': int64
        'interval': interval<locus<GRCh37>>
        'gene_type': str
        'gene_length': int32
        'exac_pLI': float64
        'exac_obs_lof': int32
        'exac_exp_lof': float64
        'exac_oe_lof': float64
        'brain_expression': float64
    ----------------------------------------
    Key: ['gene', 'transcript']
    ----------------------------------------

