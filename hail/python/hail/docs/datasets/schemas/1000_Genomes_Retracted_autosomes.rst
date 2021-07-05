.. _1000_Genomes_Retracted_autosomes:

1000_Genomes_Retracted_autosomes
================================

*  **Versions:** phase_3
*  **Reference genome builds:** GRCh38
*  **Type:** :class:`hail.MatrixTable`

Schema (phase_3, GRCh38)
~~~~~~~~~~~~~~~~~~~~~~~~

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
        's': str
        'population': str
        'super_population': str
        'is_female': bool
        'family_id': str
        'relationship_role': str
        'maternal_id': str
        'paternal_id': str
        'children_ids': array<str>
        'sibling_ids': array<str>
        'second_order_relationship_ids': array<str>
        'third_order_relationship_ids': array<str>
        'sample_qc': struct {
            call_rate: float64,
            n_called: int64,
            n_not_called: int64,
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
            CIEND: int32,
            CIPOS: int32,
            CS: str,
            END: int32,
            IMPRECISE: bool,
            MC: array<str>,
            MEINFO: array<str>,
            MEND: int32,
            MLEN: int32,
            MSTART: int32,
            SVLEN: array<int32>,
            SVTYPE: str,
            TSD: str,
            AC: int32,
            AF: float64,
            NS: int32,
            AN: int32,
            EAS_AF: float64,
            EUR_AF: float64,
            AFR_AF: float64,
            AMR_AF: float64,
            SAS_AF: float64,
            DP: int32,
            AA: str,
            VT: str,
            EX_TARGET: bool,
            MULTI_ALLELIC: bool,
            STRAND_FLIP: bool,
            REF_SWITCH: bool,
            DEPRECATED_RSID: array<str>,
            RSID_REMOVED: array<str>,
            GRCH37_38_REF_STRING_MATCH: bool,
            NOT_ALL_RSIDS_STRAND_CHANGE_OR_REF_SWITCH: bool,
            GRCH37_POS: int32,
            GRCH37_REF: str,
            ALLELE_TRANSFORM: bool,
            REF_NEW_ALLELE: bool,
            CHROM_CHANGE_BETWEEN_ASSEMBLIES: str
        }
        'a_index': int32
        'was_split': bool
        'old_locus': locus<GRCh38>
        'old_alleles': array<str>
        'variant_qc': struct {
            AC: array<int32>,
            AF: array<float64>,
            AN: int32,
            homozygote_count: array<int32>,
            n_called: int64,
            n_not_called: int64,
            call_rate: float32,
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
