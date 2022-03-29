.. _gnomad_hgdp_1kg_callset:

gnomad_hgdp_1kg_callset
=======================

*  **Versions:** 3.1
*  **Reference genome builds:** GRCh38
*  **Type:** :class:`hail.MatrixTable`

Schema (3.1, GRCh38)
~~~~~~~~~~~~~~~~~~~~

.. code-block:: text

    ----------------------------------------
    Global fields:
        'global_annotation_descriptions': struct {
            sex_imputation_ploidy_cutoffs: struct {
                Description: str
            },
            population_inference_pca_metrics: struct {
                Description: str
            },
            hard_filter_cutoffs: struct {
                Description: str
            },
            cohort_freq_meta: struct {
                Description: str
            },
            gnomad_freq_meta: struct {
                Description: str
            },
            cohort_freq_index_dict: struct {
                Description: str
            },
            gnomad_freq_index_dict: struct {
                Description: str
            },
            gnomad_faf_index_dict: struct {
                Description: str
            },
            gnomad_faf_meta: struct {
                Description: str
            },
            vep_version: struct {
                Description: str
            },
            vep_csq_header: struct {
                Description: str
            },
            dbsnp_version: struct {
                Description: str
            },
            filtering_model: struct {
                Description: str,
                sub_globals: struct {
                    model_name: struct {
                        Description: str
                    },
                    score_name: struct {
                        Description: str
                    },
                    snv_cutoff: struct {
                        Description: str,
                        sub_globals: struct {
                            bin: struct {
                                Description: str
                            },
                            min_score: struct {
                                Description: str
                            }
                        }
                    },
                    indel_cutoff: struct {
                        Description: str,
                        sub_globals: struct {
                            bin: struct {
                                Description: str
                            },
                            min_score: struct {
                                Description: str
                            }
                        }
                    },
                    snv_training_variables: struct {
                        Description: str
                    },
                    indel_training_variables: struct {
                        Description: str
                    }
                }
            },
            inbreeding_coeff_cutoff: struct {
                Description: str
            }
        }
        'sample_annotation_descriptions': struct {
            s: struct {
                Description: str
            },
            bam_metrics: struct {
                Description: str,
                sub_annotations: struct {
                    pct_bases_20x: struct {
                        Description: str
                    },
                    pct_chimeras: struct {
                        Description: str
                    },
                    freemix: struct {
                        Description: str
                    },
                    mean_coverage: struct {
                        Description: str
                    },
                    median_coverage: struct {
                        Description: str
                    },
                    mean_insert_size: struct {
                        Description: str
                    },
                    median_insert_size: struct {
                        Description: str
                    },
                    pct_bases_10x: struct {
                        Description: str
                    }
                }
            },
            subsets: struct {
                Description: str,
                sub_annotations: struct {
                    tgp: struct {
                        Description: str
                    },
                    hgdp: struct {
                        Description: str
                    }
                }
            },
            sex_imputation: struct {
                Description: str,
                sub_annotations: struct {
                    f_stat: struct {
                        Description: str
                    },
                    n_called: struct {
                        Description: str
                    },
                    expected_homs: struct {
                        Description: str
                    },
                    observed_homs: struct {
                        Description: str
                    },
                    chr20_mean_dp: struct {
                        Description: str
                    },
                    chrX_mean_dp: struct {
                        Description: str
                    },
                    chrY_mean_dp: struct {
                        Description: str
                    },
                    chrX_ploidy: struct {
                        Description: str
                    },
                    chrY_ploidy: struct {
                        Description: str
                    },
                    X_karyotype: struct {
                        Description: str
                    },
                    Y_karyotype: struct {
                        Description: str
                    },
                    sex_karyotype: struct {
                        Description: str
                    }
                }
            },
            sample_qc: struct {
                Description: str,
                sub_annotations: struct {
                    n_hom_ref: struct {
                        Description: str
                    },
                    n_het: struct {
                        Description: str
                    },
                    n_hom_var: struct {
                        Description: str
                    },
                    n_non_ref: struct {
                        Description: str
                    },
                    n_snp: struct {
                        Description: str
                    },
                    n_insertion: struct {
                        Description: str
                    },
                    n_deletion: struct {
                        Description: str
                    },
                    n_transition: struct {
                        Description: str
                    },
                    n_transversion: struct {
                        Description: str
                    },
                    r_ti_tv: struct {
                        Description: str
                    },
                    r_het_hom_var: struct {
                        Description: str
                    },
                    r_insertion_deletion: struct {
                        Description: str
                    }
                }
            },
            population_inference: struct {
                Description: str,
                sub_annotations: struct {
                    pca_scores: struct {
                        Description: str
                    },
                    pop: struct {
                        Description: str
                    },
                    prob_afr: struct {
                        Description: str
                    },
                    prob_ami: struct {
                        Description: str
                    },
                    prob_amr: struct {
                        Description: str
                    },
                    prob_asj: struct {
                        Description: str
                    },
                    prob_eas: struct {
                        Description: str
                    },
                    prob_fin: struct {
                        Description: str
                    },
                    prob_mid: struct {
                        Description: str
                    },
                    prob_nfe: struct {
                        Description: str
                    },
                    prob_oth: struct {
                        Description: str
                    },
                    prob_sas: struct {
                        Description: str
                    }
                }
            },
            labeled_subpop: struct {
                Description: str
            },
            gnomad_release: struct {
                Description: str
            }
        }
        'sex_imputation_ploidy_cutoffs': struct {
            x_ploidy_cutoffs: struct {
                upper_cutoff_X: float64,
                lower_cutoff_XX: float64,
                upper_cutoff_XX: float64,
                lower_cutoff_XXX: float64
            },
            y_ploidy_cutoffs: struct {
                lower_cutoff_Y: float64,
                upper_cutoff_Y: float64,
                lower_cutoff_YY: float64
            },
            f_stat_cutoff: float64
        }
        'population_inference_pca_metrics': struct {
            n_pcs: int32,
            min_prob: float64
        }
        'hard_filter_cutoffs': struct {
            min_cov: int32,
            max_n_snp: float64,
            min_n_snp: float64,
            max_n_singleton: float64,
            max_r_het_hom_var: float64,
            max_pct_contamination: float64,
            max_pct_chimera: float64,
            min_median_insert_size: int32
        }
        'cohort_freq_meta': array<dict<str, str>>
        'cohort_freq_index_dict': dict<str, int32>
        'gnomad_freq_meta': array<dict<str, str>>
        'gnomad_freq_index_dict': dict<str, int32>
        'gnomad_faf_index_dict': dict<str, int32>
        'gnomad_faf_meta': array<dict<str, str>>
        'vep_version': str
        'vep_csq_header': str
        'dbsnp_version': str
        'filtering_model': struct {
            model_name: str,
            score_name: str,
            snv_cutoff: struct {
                bin: float64,
                min_score: float64
            },
            indel_cutoff: struct {
                bin: float64,
                min_score: float64
            },
            snv_training_variables: array<str>,
            indel_training_variables: array<str>
        }
        'inbreeding_coeff_cutoff': float64
    ----------------------------------------
    Column fields:
        's': str
        'bam_metrics': struct {
            pct_bases_20x: float64,
            pct_chimeras: float64,
            freemix: float64,
            mean_coverage: float64,
            median_coverage: float64,
            mean_insert_size: float64,
            median_insert_size: float64,
            pct_bases_10x: float64
        }
        'subsets': struct {
            tgp: bool,
            hgdp: bool
        }
        'sex_imputation': struct {
            chr20_mean_dp: float32,
            chrX_mean_dp: float32,
            chrY_mean_dp: float32,
            chrX_ploidy: float32,
            chrY_ploidy: float32,
            X_karyotype: str,
            Y_karyotype: str,
            sex_karyotype: str,
            impute_sex_stats: struct {
                f_stat: float64,
                n_called: int64,
                expected_homs: float64,
                observed_homs: int64
            }
        }
        'sample_qc': struct {
            n_hom_ref: int64,
            n_het: int64,
            n_hom_var: int64,
            n_non_ref: int64,
            n_snp: int64,
            n_insertion: int64,
            n_deletion: int64,
            n_transition: int64,
            n_transversion: int64,
            r_ti_tv: float64,
            r_het_hom_var: float64,
            r_insertion_deletion: float64
        }
        'population_inference': struct {
            pca_scores: array<float64>,
            pop: str,
            prob_afr: float64,
            prob_ami: float64,
            prob_amr: float64,
            prob_asj: float64,
            prob_eas: float64,
            prob_fin: float64,
            prob_mid: float64,
            prob_nfe: float64,
            prob_oth: float64,
            prob_sas: float64
        }
        'labeled_subpop': str
        'gnomad_release': bool
    ----------------------------------------
    Row fields:
        'locus': locus<GRCh38>
        'alleles': array<str>
        'rsid': str
        'AS_lowqual': bool
        'telomere_or_centromere': bool
        'cohort_freq': array<struct {
            AC: int32,
            AF: float64,
            AN: int32,
            homozygote_count: int32
        }>
        'gnomad_freq': array<struct {
            AC: int32,
            AF: float64,
            AN: int32,
            homozygote_count: int32
        }>
        'gnomad_raw_qual_hists': struct {
            gq_hist_all: struct {
                bin_edges: array<float64>,
                bin_freq: array<int64>,
                n_smaller: int64,
                n_larger: int64
            },
            dp_hist_all: struct {
                bin_edges: array<float64>,
                bin_freq: array<int64>,
                n_smaller: int64,
                n_larger: int64
            },
            gq_hist_alt: struct {
                bin_edges: array<float64>,
                bin_freq: array<int64>,
                n_smaller: int64,
                n_larger: int64
            },
            dp_hist_alt: struct {
                bin_edges: array<float64>,
                bin_freq: array<int64>,
                n_smaller: int64,
                n_larger: int64
            },
            ab_hist_alt: struct {
                bin_edges: array<float64>,
                bin_freq: array<int64>,
                n_smaller: int64,
                n_larger: int64
            }
        }
        'gnomad_popmax': struct {
            AC: int32,
            AF: float64,
            AN: int32,
            homozygote_count: int32,
            pop: str,
            faf95: float64
        }
        'gnomad_qual_hists': struct {
            gq_hist_all: struct {
                bin_edges: array<float64>,
                bin_freq: array<int64>,
                n_smaller: int64,
                n_larger: int64
            },
            dp_hist_all: struct {
                bin_edges: array<float64>,
                bin_freq: array<int64>,
                n_smaller: int64,
                n_larger: int64
            },
            gq_hist_alt: struct {
                bin_edges: array<float64>,
                bin_freq: array<int64>,
                n_smaller: int64,
                n_larger: int64
            },
            dp_hist_alt: struct {
                bin_edges: array<float64>,
                bin_freq: array<int64>,
                n_smaller: int64,
                n_larger: int64
            },
            ab_hist_alt: struct {
                bin_edges: array<float64>,
                bin_freq: array<int64>,
                n_smaller: int64,
                n_larger: int64
            }
        }
        'gnomad_faf': array<struct {
            faf95: float64,
            faf99: float64
        }>
        'filters': set<str>
        'info': struct {
            QUALapprox: int64,
            SB: array<int32>,
            MQ: float64,
            MQRankSum: float64,
            VarDP: int32,
            AS_ReadPosRankSum: float64,
            AS_pab_max: float64,
            AS_QD: float32,
            AS_MQ: float64,
            QD: float32,
            AS_MQRankSum: float64,
            FS: float64,
            AS_FS: float64,
            ReadPosRankSum: float64,
            AS_QUALapprox: int64,
            AS_SB_TABLE: array<int32>,
            AS_VarDP: int32,
            AS_SOR: float64,
            SOR: float64,
            transmitted_singleton: bool,
            omni: bool,
            mills: bool,
            monoallelic: bool,
            AS_VQSLOD: float64,
            InbreedingCoeff: float32
        }
        'vep': struct {
            assembly_name: str,
            allele_string: str,
            ancestral: str,
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
                appris: str,
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
                tsl: int32,
                uniparc: str,
                variant_allele: str
            }>,
            variant_class: str
        }
        'vqsr': struct {
            AS_VQSLOD: float64,
            AS_culprit: str,
            NEGATIVE_TRAIN_SITE: bool,
            POSITIVE_TRAIN_SITE: bool
        }
        'region_flag': struct {
            lcr: bool,
            segdup: bool
        }
        'allele_info': struct {
            variant_type: str,
            allele_type: str,
            n_alt_alleles: int32,
            was_mixed: bool
        }
        'cadd': struct {
            raw_score: float32,
            phred: float32
        }
        'revel': struct {
            revel_score: float64,
            ref_aa: str,
            alt_aa: str
        }
        'splice_ai': struct {
            splice_ai: array<float32>,
            max_ds: float32,
            splice_consequence: str
        }
        'primate_ai': struct {
            primate_ai_score: float32
        }
    ----------------------------------------
    Entry fields:
        'END': int32
        'DP': int32
        'GQ': int32
        'MIN_DP': int32
        'PID': str
        'RGQ': int32
        'SB': array<int32>
        'GT': call
        'PGT': call
        'AD': array<int32>
        'PL': array<int32>
    ----------------------------------------
    Column key: ['s']
    Row key: ['locus', 'alleles']
    ----------------------------------------

