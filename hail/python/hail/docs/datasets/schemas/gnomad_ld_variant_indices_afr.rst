.. _gnomad_ld_variant_indices_afr:

gnomad_ld_variant_indices_afr
=============================

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
        'age_distribution': array<int32>
    ----------------------------------------
    Row fields:
        'locus': locus<GRCh37>
        'alleles': array<str>
        'pop_freq': struct {
            AC: int32,
            AF: float64,
            AN: int32,
            homozygote_count: int32
        }
        'idx': int64
    ----------------------------------------
    Key: ['locus', 'alleles']
    ----------------------------------------

