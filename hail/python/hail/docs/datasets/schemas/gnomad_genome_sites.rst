.. _gnomad_genome_sites:

gnomad_genome_sites
===================

*  **Versions:** 2.1.1, 3.1
*  **Reference genome builds:** GRCh37, GRCh38
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
        'freq': array<struct {
            AC: int32,
            AF: float64,
            AN: int32,
            homozygote_count: int32
        }>
        'age_hist_het': array<struct {
            bin_edges: array<float64>,
            bin_freq: array<int64>,
            n_smaller: int64,
            n_larger: int64
        }>
        'age_hist_hom': array<struct {
            bin_edges: array<float64>,
            bin_freq: array<int64>,
            n_smaller: int64,
            n_larger: int64
        }>
        'popmax': array<struct {
            AC: int32,
            AF: float64,
            AN: int32,
            homozygote_count: int32,
            pop: str
        }>
        'faf': array<struct {
            meta: dict<str, str>,
            faf95: float64,
            faf99: float64
        }>
        'lcr': bool
        'decoy': bool
        'segdup': bool
        'nonpar': bool
        'variant_type': str
        'allele_type': str
        'n_alt_alleles': int32
        'was_mixed': bool
        'has_star': bool
        'qd': float64
        'pab_max': float64
        'info_MQRankSum': float64
        'info_SOR': float64
        'info_InbreedingCoeff': float64
        'info_ReadPosRankSum': float64
        'info_FS': float64
        'info_QD': float64
        'info_MQ': float64
        'info_DP': int32
        'transmitted_singleton': bool
        'fail_hard_filters': bool
        'info_POSITIVE_TRAIN_SITE': bool
        'info_NEGATIVE_TRAIN_SITE': bool
        'omni': bool
        'mills': bool
        'tp': bool
        'rf_train': bool
        'rf_label': str
        'rf_probability': float64
        'rank': int64
        'was_split': bool
        'singleton': bool
        '_score': float64
        '_singleton': bool
        'biallelic_rank': int64
        'singleton_rank': int64
        'n_nonref': int32
        'score': float64
        'adj_biallelic_singleton_rank': int64
        'adj_rank': int64
        'adj_biallelic_rank': int64
        'adj_singleton_rank': int64
        'biallelic_singleton_rank': int64
        'filters': set<str>
        'gq_hist_alt': struct {
            bin_edges: array<float64>,
            bin_freq: array<int64>,
            n_smaller: int64,
            n_larger: int64
        }
        'gq_hist_all': struct {
            bin_edges: array<float64>,
            bin_freq: array<int64>,
            n_smaller: int64,
            n_larger: int64
        }
        'dp_hist_alt': struct {
            bin_edges: array<float64>,
            bin_freq: array<int64>,
            n_smaller: int64,
            n_larger: int64
        }
        'dp_hist_all': struct {
            bin_edges: array<float64>,
            bin_freq: array<int64>,
            n_smaller: int64,
            n_larger: int64
        }
        'ab_hist_alt': struct {
            bin_edges: array<float64>,
            bin_freq: array<int64>,
            n_smaller: int64,
            n_larger: int64
        }
        'qual': float64
        'vep': struct {
            assembly_name: str,
            allele_string: str,
            ancestral: str,
            colocated_variants: array<struct {
                aa_allele: str,
                aa_maf: float64,
                afr_allele: str,
                afr_maf: float64,
                allele_string: str,
                amr_allele: str,
                amr_maf: float64,
                clin_sig: array<str>,
                end: int32,
                eas_allele: str,
                eas_maf: float64,
                ea_allele: str,
                ea_maf: float64,
                eur_allele: str,
                eur_maf: float64,
                exac_adj_allele: str,
                exac_adj_maf: float64,
                exac_allele: str,
                exac_afr_allele: str,
                exac_afr_maf: float64,
                exac_amr_allele: str,
                exac_amr_maf: float64,
                exac_eas_allele: str,
                exac_eas_maf: float64,
                exac_fin_allele: str,
                exac_fin_maf: float64,
                exac_maf: float64,
                exac_nfe_allele: str,
                exac_nfe_maf: float64,
                exac_oth_allele: str,
                exac_oth_maf: float64,
                exac_sas_allele: str,
                exac_sas_maf: float64,
                id: str,
                minor_allele: str,
                minor_allele_freq: float64,
                phenotype_or_disease: int32,
                pubmed: array<int32>,
                sas_allele: str,
                sas_maf: float64,
                somatic: int32,
                start: int32,
                strand: int32
            }>,
            context: str,
            end: int32,
            id: str,
            input: str,
            intergenic_consequences: array<struct {
                allele_num: int32,
                consequence_terms: array<str>,
                impact: str,
                minimised: int32,
                variant_allele: str
            }>,
            most_severe_consequence: str,
            motif_feature_consequences: array<struct {
                allele_num: int32,
                consequence_terms: array<str>,
                high_inf_pos: str,
                impact: str,
                minimised: int32,
                motif_feature_id: str,
                motif_name: str,
                motif_pos: int32,
                motif_score_change: float64,
                strand: int32,
                variant_allele: str
            }>,
            regulatory_feature_consequences: array<struct {
                allele_num: int32,
                biotype: str,
                consequence_terms: array<str>,
                impact: str,
                minimised: int32,
                regulatory_feature_id: str,
                variant_allele: str
            }>,
            seq_region_name: str,
            start: int32,
            strand: int32,
            transcript_consequences: array<struct {
                allele_num: int32,
                amino_acids: str,
                biotype: str,
                canonical: int32,
                ccds: str,
                cdna_start: int32,
                cdna_end: int32,
                cds_end: int32,
                cds_start: int32,
                codons: str,
                consequence_terms: array<str>,
                distance: int32,
                domains: array<struct {
                    db: str,
                    name: str
                }>,
                exon: str,
                gene_id: str,
                gene_pheno: int32,
                gene_symbol: str,
                gene_symbol_source: str,
                hgnc_id: str,
                hgvsc: str,
                hgvsp: str,
                hgvs_offset: int32,
                impact: str,
                intron: str,
                lof: str,
                lof_flags: str,
                lof_filter: str,
                lof_info: str,
                minimised: int32,
                polyphen_prediction: str,
                polyphen_score: float64,
                protein_end: int32,
                protein_start: int32,
                protein_id: str,
                sift_prediction: str,
                sift_score: float64,
                strand: int32,
                swissprot: str,
                transcript_id: str,
                trembl: str,
                uniparc: str,
                variant_allele: str
            }>,
            variant_class: str
        }
        'allele_info': struct {
            BaseQRankSum: float64,
            ClippingRankSum: float64,
            DB: bool,
            DP: int32,
            DS: bool,
            END: int32,
            FS: float64,
            HaplotypeScore: float64,
            InbreedingCoeff: float64,
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
            culprit: str
        }
        'rsid': str
    ----------------------------------------
    Key: ['locus', 'alleles']
    ----------------------------------------

