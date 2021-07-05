.. _1000_Genomes_HighCov_chrY:

1000_Genomes_HighCov_chrY
=========================

*  **Versions:** NYGC_30x_unphased
*  **Reference genome builds:** GRCh38
*  **Type:** :class:`hail.MatrixTable`

Schema (NYGC_30x_unphased, GRCh38)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

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
        's': str
        'FamilyID': str
        'FatherID': str
        'MotherID': str
        'Sex': str
        'Population': str
        'Superpopulation': str
        'sample_qc': struct {
            dp_stats: struct {
                mean: float64,
                stdev: float64,
                min: float64,
                max: float64
            },
            gq_stats: struct {
                mean: float64,
                stdev: float64,
                min: float64,
                max: float64
            },
            call_rate: float64,
            n_called: int64,
            n_not_called: int64,
            n_filtered: int64,
            n_hom_ref: int64,
            n_het: int64,
            n_hom_var: int64,
            n_non_ref: int64,
            n_singleton: int64,
            n_snp: int64,
            n_insertion: int64,
            n_deletion: int64,
            n_transition: int64,
            n_transversion: int64,
            n_star: int64,
            r_ti_tv: float64,
            r_het_hom_var: float64,
            r_insertion_deletion: float64
        }
    ----------------------------------------
    Row fields:
        'locus': locus<GRCh38>
        'alleles': array<str>
        'rsid': str
        'qual': float64
        'filters': set<str>
        'info': struct {
            AC: int32,
            AF: float64,
            AN: int32,
            BaseQRankSum: float64,
            ClippingRankSum: float64,
            DP: int32,
            DS: bool,
            END: int32,
            ExcessHet: float64,
            FS: float64,
            HaplotypeScore: float64,
            InbreedingCoeff: float64,
            MLEAC: int32,
            MLEAF: float64,
            MQ: float64,
            MQ0: int32,
            MQRankSum: float64,
            NEGATIVE_TRAIN_SITE: bool,
            POSITIVE_TRAIN_SITE: bool,
            QD: float64,
            RAW_MQ: float64,
            ReadPosRankSum: float64,
            SOR: float64,
            VQSLOD: float64,
            VariantType: str,
            culprit: str,
            AN_EAS: int32,
            AN_AMR: int32,
            AN_EUR: int32,
            AN_AFR: int32,
            AN_SAS: int32,
            AN_EUR_unrel: int32,
            AN_EAS_unrel: int32,
            AN_AMR_unrel: int32,
            AN_SAS_unrel: int32,
            AN_AFR_unrel: int32,
            AC_EAS: int32,
            AC_AMR: int32,
            AC_EUR: int32,
            AC_AFR: int32,
            AC_SAS: int32,
            AC_EUR_unrel: int32,
            AC_EAS_unrel: int32,
            AC_AMR_unrel: int32,
            AC_SAS_unrel: int32,
            AC_AFR_unrel: int32,
            AF_EAS: float64,
            AF_AMR: float64,
            AF_EUR: float64,
            AF_AFR: float64,
            AF_SAS: float64,
            AF_EUR_unrel: float64,
            AF_EAS_unrel: float64,
            AF_AMR_unrel: float64,
            AF_SAS_unrel: float64,
            AF_AFR_unrel: float64
        }
        'a_index': int32
        'was_split': bool
        'variant_qc': struct {
            dp_stats: struct {
                mean: float64,
                stdev: float64,
                min: float64,
                max: float64
            },
            gq_stats: struct {
                mean: float64,
                stdev: float64,
                min: float64,
                max: float64
            },
            AC: array<int32>,
            AF: array<float64>,
            AN: int32,
            homozygote_count: array<int32>,
            call_rate: float64,
            n_called: int64,
            n_not_called: int64,
            n_filtered: int64,
            n_het: int64,
            n_non_ref: int64,
            het_freq_hwe: float64,
            p_value_hwe: float64
        }
    ----------------------------------------
    Entry fields:
        'AB': float64
        'AD': array<int32>
        'DP': int32
        'GQ': int32
        'GT': call
        'MIN_DP': int32
        'MQ0': int32
        'PGT': call
        'PID': str
        'PL': array<int32>
        'RGQ': int32
        'SB': array<int32>
    ----------------------------------------
    Column key: ['s']
    Row key: ['locus', 'alleles']
    ----------------------------------------
