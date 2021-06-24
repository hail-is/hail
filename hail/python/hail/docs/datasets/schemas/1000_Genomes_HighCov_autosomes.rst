.. _1000_Genomes_HighCov_autosomes:

1000_Genomes_HighCov_autosomes
==============================

*  **Versions:** NYGC_30x
*  **Reference genome builds:** GRCh38
*  **Type:** :class:`hail.MatrixTable`

Schema (NYGC_30x, GRCh38)
~~~~~~~~~~~~~~~~~~~~~~~~~

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
            AC: array<int32>,
            AC_AFR: array<int32>,
            AC_AMR: array<int32>,
            AC_EAS: array<int32>,
            AC_EUR: array<int32>,
            AC_Het: array<int32>,
            AC_Het_AFR: array<int32>,
            AC_Het_AMR: array<int32>,
            AC_Het_EAS: array<int32>,
            AC_Het_EUR: array<int32>,
            AC_Het_SAS: array<int32>,
            AC_Hom: array<int32>,
            AC_Hom_AFR: array<int32>,
            AC_Hom_AMR: array<int32>,
            AC_Hom_EAS: array<int32>,
            AC_Hom_EUR: array<int32>,
            AC_Hom_SAS: array<int32>,
            AC_SAS: array<int32>,
            AF: array<float64>,
            AF_AFR: array<float64>,
            AF_AMR: array<float64>,
            AF_EAS: array<float64>,
            AF_EUR: array<float64>,
            AF_SAS: array<float64>,
            AN: int32,
            AN_AFR: int32,
            AN_AMR: int32,
            AN_EAS: int32,
            AN_EUR: int32,
            AN_SAS: int32,
            BaseQRankSum: float64,
            ClippingRankSum: float64,
            DP: int32,
            DS: bool,
            END: int32,
            FS: float64,
            HWE: array<float64>,
            HWE_AFR: array<float64>,
            HWE_AMR: array<float64>,
            HWE_EAS: array<float64>,
            HWE_EUR: array<float64>,
            HWE_SAS: array<float64>,
            HaplotypeScore: float64,
            InbreedingCoeff: float64,
            MLEAC: array<int32>,
            MLEAF: array<float64>,
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
            AN_EUR_unrel: int32,
            AN_EAS_unrel: int32,
            AN_AMR_unrel: int32,
            AN_SAS_unrel: int32,
            AN_AFR_unrel: int32,
            AC_EUR_unrel: array<int32>,
            AC_EAS_unrel: array<int32>,
            AC_AMR_unrel: array<int32>,
            AC_SAS_unrel: array<int32>,
            AC_AFR_unrel: array<int32>,
            AC_Hom_EUR_unrel: array<int32>,
            AC_Hom_EAS_unrel: array<int32>,
            AC_Hom_AMR_unrel: array<int32>,
            AC_Hom_SAS_unrel: array<int32>,
            AC_Hom_AFR_unrel: array<int32>,
            AC_Het_EUR_unrel: array<int32>,
            AC_Het_EAS_unrel: array<int32>,
            AC_Het_AMR_unrel: array<int32>,
            AC_Het_SAS_unrel: array<int32>,
            AC_Het_AFR_unrel: array<int32>,
            AF_EUR_unrel: array<float64>,
            AF_EAS_unrel: array<float64>,
            AF_AMR_unrel: array<float64>,
            AF_SAS_unrel: array<float64>,
            AF_AFR_unrel: array<float64>,
            ExcHet_EAS: array<float64>,
            ExcHet_AMR: array<float64>,
            ExcHet_EUR: array<float64>,
            ExcHet_AFR: array<float64>,
            ExcHet_SAS: array<float64>,
            ExcHet: array<float64>
        }
        'variant_qc': struct {
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
        'GT': call
    ----------------------------------------
    Column key: ['s']
    Row key: ['locus', 'alleles']
    ----------------------------------------
