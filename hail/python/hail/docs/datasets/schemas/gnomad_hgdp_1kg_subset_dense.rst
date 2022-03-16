.. _gnomad_hgdp_1kg_subset_dense:

gnomad_hgdp_1kg_subset_dense
============================

*  **Versions:** 3.1, 3.1.2
*  **Reference genome builds:** GRCh38
*  **Type:** :class:`hail.MatrixTable`

Schema (3.1.2, GRCh38)
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: text

    ----------------------------------------
    Global fields:
        'global_annotation_descriptions': struct {
            gnomad_sex_imputation_ploidy_cutoffs: struct {
                Description: str
            },
            gnomad_population_inference_pca_metrics: struct {
                Description: str
            },
            sample_hard_filter_cutoffs: struct {
                Description: str
            },
            gnomad_sample_qc_metric_outlier_cutoffs: struct {
                Description: str
            },
            gnomad_age_distribution: struct {
                Description: str,
                sub_globals: struct {
                    bin_edges: struct {
                        Description: str
                    },
                    bin_freq: struct {
                        Description: str
                    },
                    n_smaller: struct {
                        Description: str
                    },
                    n_larger: struct {
                        Description: str
                    }
                }
            },
            hgdp_tgp_freq_meta: struct {
                Description: str
            },
            gnomad_freq_meta: struct {
                Description: str
            },
            hgdp_tgp_freq_index_dict: struct {
                Description: str
            },
            gnomad_freq_index_dict: struct {
                Description: str
            },
            gnomad_faf_meta: struct {
                Description: str
            },
            gnomad_faf_index_dict: struct {
                Description: str
            },
            variant_filtering_model: struct {
                Description: set<str>,
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
            variant_inbreeding_coeff_cutoff: struct {
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
            sample_qc: struct {
                Description: str,
                sub_annotations: struct {
                    n_deletion: struct {
                        Description: str
                    },
                    n_het: struct {
                        Description: str
                    },
                    n_hom_ref: struct {
                        Description: str
                    },
                    n_hom_var: struct {
                        Description: str
                    },
                    n_insertion: struct {
                        Description: str
                    },
                    n_non_ref: struct {
                        Description: str
                    },
                    n_snp: struct {
                        Description: str
                    },
                    n_transition: struct {
                        Description: str
                    },
                    n_transversion: struct {
                        Description: str
                    },
                    r_het_hom_var: struct {
                        Description: str
                    },
                    r_insertion_deletion: struct {
                        Description: str
                    },
                    r_ti_tv: struct {
                        Description: str
                    }
                }
            },
            gnomad_sex_imputation: struct {
                Description: str,
                sub_annotations: struct {
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
                    },
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
                    }
                }
            },
            gnomad_population_inference: struct {
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
            gnomad_sample_qc_residuals: struct {
                Description: tuple (
                    str
                ),
                sub_annotations: struct {
                    n_snp_residual: struct {
                        Description: str
                    },
                    r_ti_tv_residual: struct {
                        Description: str
                    },
                    r_insertion_deletion_residual: struct {
                        Description: str
                    },
                    n_insertion_residual: struct {
                        Description: str
                    },
                    n_deletion_residual: struct {
                        Description: str
                    },
                    r_het_hom_var_residual: struct {
                        Description: str
                    },
                    n_transition_residual: struct {
                        Description: str
                    },
                    n_transversion_residual: struct {
                        Description: str
                    }
                }
            },
            gnomad_sample_filters: struct {
                Description: str,
                sub_annotations: struct {
                    hard_filters: struct {
                        Description: str
                    },
                    hard_filtered: struct {
                        Description: str
                    },
                    release_related: struct {
                        Description: str
                    },
                    qc_metrics_filters: struct {
                        Description: str
                    }
                }
            },
            gnomad_high_quality: struct {
                Description: str
            },
            gnomad_release: struct {
                Description: str
            },
            relatedness_inference: struct {
                Description: str,
                sub_annotations: struct {
                    related_samples: struct {
                        Description: str,
                        sub_annotations: struct {
                            s: struct {
                                Description: str
                            },
                            kin: struct {
                                Description: str
                            },
                            ibd0: struct {
                                Description: str
                            },
                            ibd1: struct {
                                Description: str
                            },
                            ibd2: struct {
                                Description: str
                            }
                        }
                    },
                    related: struct {
                        Description: str
                    }
                }
            },
            hgdp_tgp_meta: struct {
                Description: str,
                sub_annotations: struct {
                    project: struct {
                        Description: str
                    },
                    study_region: struct {
                        Description: str
                    },
                    population: struct {
                        Description: str
                    },
                    genetic_region: struct {
                        Description: str
                    },
                    latitude: struct {
                        Description: str
                    },
                    longitude: struct {
                        Description: str
                    },
                    hgdp_technical_meta: struct {
                        Description: str,
                        sub_annotations: struct {
                            source: struct {
                                Description: str
                            },
                            library_type: struct {
                                Description: str
                            }
                        }
                    },
                    global_pca_scores: struct {
                        Description: str
                    },
                    subcontinental_pca: struct {
                        Description: str,
                        sub_annotations: struct {
                            pca_scores: struct {
                                Description: str
                            },
                            pca_scores_outliers_removed: struct {
                                Description: str
                            },
                            outlier: struct {
                                Description: str
                            }
                        }
                    },
                    gnomad_labeled_subpop: struct {
                        Description: str
                    }
                }
            },
            high_quality: struct {
                Description: str
            }
        }
        'gnomad_sex_imputation_ploidy_cutoffs': struct {
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
        'gnomad_population_inference_pca_metrics': struct {
            n_pcs: int32,
            min_prob: float64
        }
        'sample_hard_filter_cutoffs': struct {
            min_cov: int32,
            max_n_snp: float64,
            min_n_snp: float64,
            max_n_singleton: float64,
            max_r_het_hom_var: float64,
            max_pct_contamination: float64,
            max_pct_chimera: float64,
            min_median_insert_size: int32
        }
        'gnomad_sample_qc_metric_outlier_cutoffs': struct {
            lms: struct {
                n_snp: struct {
                    beta: array<float64>,
                    standard_error: array<float64>,
                    t_stat: array<float64>,
                    p_value: array<float64>,
                    multiple_standard_error: float64,
                    multiple_r_squared: float64,
                    adjusted_r_squared: float64,
                    f_stat: float64,
                    multiple_p_value: float64,
                    n: int32
                },
                n_singleton: struct {
                    beta: array<float64>,
                    standard_error: array<float64>,
                    t_stat: array<float64>,
                    p_value: array<float64>,
                    multiple_standard_error: float64,
                    multiple_r_squared: float64,
                    adjusted_r_squared: float64,
                    f_stat: float64,
                    multiple_p_value: float64,
                    n: int32
                },
                r_ti_tv: struct {
                    beta: array<float64>,
                    standard_error: array<float64>,
                    t_stat: array<float64>,
                    p_value: array<float64>,
                    multiple_standard_error: float64,
                    multiple_r_squared: float64,
                    adjusted_r_squared: float64,
                    f_stat: float64,
                    multiple_p_value: float64,
                    n: int32
                },
                r_insertion_deletion: struct {
                    beta: array<float64>,
                    standard_error: array<float64>,
                    t_stat: array<float64>,
                    p_value: array<float64>,
                    multiple_standard_error: float64,
                    multiple_r_squared: float64,
                    adjusted_r_squared: float64,
                    f_stat: float64,
                    multiple_p_value: float64,
                    n: int32
                },
                n_insertion: struct {
                    beta: array<float64>,
                    standard_error: array<float64>,
                    t_stat: array<float64>,
                    p_value: array<float64>,
                    multiple_standard_error: float64,
                    multiple_r_squared: float64,
                    adjusted_r_squared: float64,
                    f_stat: float64,
                    multiple_p_value: float64,
                    n: int32
                },
                n_deletion: struct {
                    beta: array<float64>,
                    standard_error: array<float64>,
                    t_stat: array<float64>,
                    p_value: array<float64>,
                    multiple_standard_error: float64,
                    multiple_r_squared: float64,
                    adjusted_r_squared: float64,
                    f_stat: float64,
                    multiple_p_value: float64,
                    n: int32
                },
                r_het_hom_var: struct {
                    beta: array<float64>,
                    standard_error: array<float64>,
                    t_stat: array<float64>,
                    p_value: array<float64>,
                    multiple_standard_error: float64,
                    multiple_r_squared: float64,
                    adjusted_r_squared: float64,
                    f_stat: float64,
                    multiple_p_value: float64,
                    n: int32
                },
                n_transition: struct {
                    beta: array<float64>,
                    standard_error: array<float64>,
                    t_stat: array<float64>,
                    p_value: array<float64>,
                    multiple_standard_error: float64,
                    multiple_r_squared: float64,
                    adjusted_r_squared: float64,
                    f_stat: float64,
                    multiple_p_value: float64,
                    n: int32
                },
                n_transversion: struct {
                    beta: array<float64>,
                    standard_error: array<float64>,
                    t_stat: array<float64>,
                    p_value: array<float64>,
                    multiple_standard_error: float64,
                    multiple_r_squared: float64,
                    adjusted_r_squared: float64,
                    f_stat: float64,
                    multiple_p_value: float64,
                    n: int32
                }
            },
            qc_metrics_stats: struct {
                n_snp_residual: struct {
                    median: float64,
                    mad: float64,
                    lower: float64,
                    upper: float64
                },
                n_singleton_residual: struct {
                    median: float64,
                    mad: float64,
                    lower: float64,
                    upper: float64
                },
                r_ti_tv_residual: struct {
                    median: float64,
                    mad: float64,
                    lower: float64,
                    upper: float64
                },
                r_insertion_deletion_residual: struct {
                    median: float64,
                    mad: float64,
                    lower: float64,
                    upper: float64
                },
                n_insertion_residual: struct {
                    median: float64,
                    mad: float64,
                    lower: float64,
                    upper: float64
                },
                n_deletion_residual: struct {
                    median: float64,
                    mad: float64,
                    lower: float64,
                    upper: float64
                },
                r_het_hom_var_residual: struct {
                    median: float64,
                    mad: float64,
                    lower: float64,
                    upper: float64
                },
                n_transition_residual: struct {
                    median: float64,
                    mad: float64,
                    lower: float64,
                    upper: float64
                },
                n_transversion_residual: struct {
                    median: float64,
                    mad: float64,
                    lower: float64,
                    upper: float64
                }
            },
            n_pcs: int32,
            used_regressed_metrics: bool
        }
        'gnomad_age_distribution': struct {
            bin_edges: array<float64>,
            bin_freq: array<int32>,
            n_smaller: int32,
            n_larger: int32
        }
        'variant_annotation_descriptions': struct {
            locus: struct {
                Description: str
            },
            alleles: struct {
                Description: str
            },
            rsid: struct {
                Description: str
            },
            a_index: struct {
                Description: str
            },
            was_split: struct {
                Description: str
            },
            hgdp_tgp_freq: struct {
                Description: str,
                sub_annotations: struct {
                    AC: struct {
                        Description: str
                    },
                    AF: struct {
                        Description: str
                    },
                    AN: struct {
                        Description: str
                    },
                    homozygote_count: struct {
                        Description: str
                    }
                }
            },
            gnomad_freq: struct {
                Description: str,
                sub_annotations: struct {
                    AC: struct {
                        Description: str
                    },
                    AF: struct {
                        Description: str
                    },
                    AN: struct {
                        Description: str
                    },
                    homozygote_count: struct {
                        Description: str
                    }
                }
            },
            gnomad_popmax: struct {
                Description: str,
                sub_annotations: struct {
                    AC: struct {
                        Description: str
                    },
                    AF: struct {
                        Description: str
                    },
                    AN: struct {
                        Description: str
                    },
                    homozygote_count: struct {
                        Description: str
                    },
                    pop: struct {
                        Description: str
                    },
                    faf95: struct {
                        Description: str
                    }
                }
            },
            gnomad_faf: struct {
                Description: str,
                sub_annotations: struct {
                    faf95: struct {
                        Description: str
                    },
                    faf99: struct {
                        Description: str
                    }
                }
            },
            gnomad_qual_hists: struct {
                Description: str,
                sub_annotations: struct {
                    gq_hist_all: struct {
                        Description: str,
                        sub_annotations: struct {
                            bin_edges: struct {
                                Description: str
                            },
                            bin_freq: struct {
                                Description: str
                            },
                            n_smaller: struct {
                                Description: str
                            },
                            n_larger: struct {
                                Description: str
                            }
                        }
                    },
                    dp_hist_all: struct {
                        Description: str,
                        sub_annotations: struct {
                            bin_edges: struct {
                                Description: str
                            },
                            bin_freq: struct {
                                Description: str
                            },
                            n_smaller: struct {
                                Description: str
                            },
                            n_larger: struct {
                                Description: str
                            }
                        }
                    },
                    gq_hist_alt: struct {
                        Description: str,
                        sub_annotations: struct {
                            bin_edges: struct {
                                Description: str
                            },
                            bin_freq: struct {
                                Description: str
                            },
                            n_smaller: struct {
                                Description: str
                            },
                            n_larger: struct {
                                Description: str
                            }
                        }
                    },
                    dp_hist_alt: struct {
                        Description: str,
                        sub_annotations: struct {
                            bin_edges: struct {
                                Description: str
                            },
                            bin_freq: struct {
                                Description: str
                            },
                            n_smaller: struct {
                                Description: str
                            },
                            n_larger: struct {
                                Description: str
                            }
                        }
                    },
                    ab_hist_alt: struct {
                        Description: str,
                        sub_annotations: struct {
                            bin_edges: struct {
                                Description: str
                            },
                            bin_freq: struct {
                                Description: str
                            },
                            n_smaller: struct {
                                Description: str
                            },
                            n_larger: struct {
                                Description: str
                            }
                        }
                    }
                }
            },
            gnomad_raw_qual_hists: struct {
                Description: str,
                sub_annotations: struct {
                    gq_hist_all: struct {
                        Description: str,
                        sub_annotations: struct {
                            bin_edges: struct {
                                Description: str
                            },
                            bin_freq: struct {
                                Description: str
                            },
                            n_smaller: struct {
                                Description: str
                            },
                            n_larger: struct {
                                Description: str
                            }
                        }
                    },
                    dp_hist_all: struct {
                        Description: str,
                        sub_annotations: struct {
                            bin_edges: struct {
                                Description: str
                            },
                            bin_freq: struct {
                                Description: str
                            },
                            n_smaller: struct {
                                Description: str
                            },
                            n_larger: struct {
                                Description: str
                            }
                        }
                    },
                    gq_hist_alt: struct {
                        Description: str,
                        sub_annotations: struct {
                            bin_edges: struct {
                                Description: str
                            },
                            bin_freq: struct {
                                Description: str
                            },
                            n_smaller: struct {
                                Description: str
                            },
                            n_larger: struct {
                                Description: str
                            }
                        }
                    },
                    dp_hist_alt: struct {
                        Description: str,
                        sub_annotations: struct {
                            bin_edges: struct {
                                Description: str
                            },
                            bin_freq: struct {
                                Description: str
                            },
                            n_smaller: struct {
                                Description: str
                            },
                            n_larger: struct {
                                Description: str
                            }
                        }
                    },
                    ab_hist_alt: struct {
                        Description: str,
                        sub_annotations: struct {
                            bin_edges: struct {
                                Description: str
                            },
                            bin_freq: struct {
                                Description: str
                            },
                            n_smaller: struct {
                                Description: str
                            },
                            n_larger: struct {
                                Description: str
                            }
                        }
                    }
                }
            },
            gnomad_age_hist_het: struct {
                Description: str,
                sub_annotations: struct {
                    bin_edges: struct {
                        Description: str
                    },
                    bin_freq: struct {
                        Description: str
                    },
                    n_smaller: struct {
                        Description: str
                    },
                    n_larger: struct {
                        Description: str
                    }
                }
            },
            gnomad_age_hist_hom: struct {
                Description: str,
                sub_annotations: struct {
                    bin_edges: struct {
                        Description: str
                    },
                    bin_freq: struct {
                        Description: str
                    },
                    n_smaller: struct {
                        Description: str
                    },
                    n_larger: struct {
                        Description: str
                    }
                }
            },
            filters: struct {
                Description: str
            },
            info: struct {
                Description: str,
                sub_annotations: struct {
                    QUALapprox: struct {
                        Description: str
                    },
                    SB: struct {
                        Description: str
                    },
                    MQ: struct {
                        Description: str
                    },
                    MQRankSum: struct {
                        Description: str
                    },
                    VarDP: struct {
                        Description: str
                    },
                    AS_ReadPosRankSum: struct {
                        Description: str
                    },
                    AS_pab_max: struct {
                        Description: str
                    },
                    AS_QD: struct {
                        Description: str
                    },
                    AS_MQ: struct {
                        Description: str
                    },
                    QD: struct {
                        Description: str
                    },
                    AS_MQRankSum: struct {
                        Description: str
                    },
                    FS: struct {
                        Description: str
                    },
                    AS_FS: struct {
                        Description: str
                    },
                    ReadPosRankSum: struct {
                        Description: str
                    },
                    AS_QUALapprox: struct {
                        Description: str
                    },
                    AS_SB_TABLE: struct {
                        Description: str
                    },
                    AS_VarDP: struct {
                        Description: str
                    },
                    AS_SOR: struct {
                        Description: str
                    },
                    SOR: struct {
                        Description: str
                    },
                    transmitted_singleton: struct {
                        Description: str
                    },
                    omni: struct {
                        Description: str
                    },
                    mills: struct {
                        Description: str
                    },
                    monoallelic: struct {
                        Description: str
                    },
                    InbreedingCoeff: struct {
                        Description: str
                    }
                }
            },
            vep: struct {
                Description: str
            },
            vqsr: struct {
                Description: str,
                sub_annotations: struct {
                    AS_VQSLOD: struct {
                        Description: str
                    },
                    AS_culprit: struct {
                        Description: str
                    },
                    NEGATIVE_TRAIN_SITE: struct {
                        Description: str
                    },
                    POSITIVE_TRAIN_SITE: struct {
                        Description: str
                    }
                }
            },
            region_flag: struct {
                Description: str,
                sub_annotations: struct {
                    lcr: struct {
                        Description: str
                    },
                    segdup: struct {
                        Description: str
                    }
                }
            },
            allele_info: struct {
                Description: str,
                sub_annotations: struct {
                    variant_type: struct {
                        Description: str
                    },
                    allele_type: struct {
                        Description: str
                    },
                    n_alt_alleles: struct {
                        Description: str
                    }
                }
            },
            was_mixed: struct {
                Description: str
            },
            cadd: struct {
                sub_annotations: struct {
                    raw_score: struct {
                        Description: str
                    },
                    phred: struct {
                        Description: str
                    },
                    has_duplicate: struct {
                        Description: str
                    }
                }
            },
            revel: struct {
                Description: str,
                sub_annotations: struct {
                    revel_score: struct {
                        Description: str
                    },
                    has_duplicate: struct {
                        Description: str
                    }
                }
            },
            splice_ai: struct {
                sub_annotations: struct {
                    splice_ai: struct {
                        Description: str
                    },
                    splice_consequence: struct {
                        Description: str
                    },
                    has_duplicate: struct {
                        Description: str
                    }
                }
            },
            primate_ai: struct {
                sub_annotations: struct {
                    primate_ai_score: struct {
                        Description: str
                    },
                    has_duplicate: struct {
                        Description: str
                    }
                }
            },
            AS_lowqual: struct {
                Description: str
            },
            telomere_or_centromere: struct {
                Description: str
            }
        }
        'hgdp_tgp_freq_meta': array<dict<str, str>>
        'hgdp_tgp_freq_index_dict': dict<str, int32>
        'gnomad_freq_meta': array<dict<str, str>>
        'gnomad_freq_index_dict': dict<str, int32>
        'gnomad_faf_index_dict': dict<str, int32>
        'gnomad_faf_meta': array<dict<str, str>>
        'vep_version': str
        'vep_csq_header': str
        'dbsnp_version': str
        'variant_filtering_model': struct {
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
        'variant_inbreeding_coeff_cutoff': float64
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
        'sample_qc': struct {
            n_deletion: int64,
            n_het: int64,
            n_hom_ref: int64,
            n_hom_var: int64,
            n_insertion: int64,
            n_non_ref: int64,
            n_snp: int64,
            n_transition: int64,
            n_transversion: int64,
            r_het_hom_var: float64,
            r_insertion_deletion: float64,
            r_ti_tv: float64
        }
        'gnomad_sex_imputation': struct {
            chr20_mean_dp: float32,
            chrX_mean_dp: float32,
            chrY_mean_dp: float32,
            chrX_ploidy: float32,
            chrY_ploidy: float32,
            X_karyotype: str,
            Y_karyotype: str,
            sex_karyotype: str,
            f_stat: float64,
            n_called: int64,
            expected_homs: float64,
            observed_homs: int64
        }
        'gnomad_population_inference': struct {
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
        'gnomad_sample_qc_residuals': struct {
            n_snp_residual: float64,
            r_ti_tv_residual: float64,
            r_insertion_deletion_residual: float64,
            n_insertion_residual: float64,
            n_deletion_residual: float64,
            r_het_hom_var_residual: float64,
            n_transition_residual: float64,
            n_transversion_residual: float64
        }
        'gnomad_sample_filters': struct {
            hard_filters: set<str>,
            hard_filtered: bool,
            release_related: bool,
            qc_metrics_filters: set<str>
        }
        'gnomad_high_quality': bool
        'gnomad_release': bool
        'relatedness_inference': struct {
            related_samples: set<struct {
                s: str,
                kin: float64,
                ibd0: float64,
                ibd1: float64,
                ibd2: float64
            }>,
            related: bool
        }
        'hgdp_tgp_meta': struct {
            project: str,
            study_region: str,
            population: str,
            genetic_region: str,
            latitude: float64,
            longitude: float64,
            hgdp_technical_meta: struct {
                source: str,
                library_type: str
            },
            global_pca_scores: array<float64>,
            subcontinental_pca: struct {
                pca_scores: array<float64>,
                pca_scores_outliers_removed: array<float64>,
                outlier: bool
            },
            gnomad_labeled_subpop: str
        }
        'high_quality': bool
    ----------------------------------------
    Row fields:
        'locus': locus<GRCh38>
        'alleles': array<str>
        'rsid': set<str>
        'a_index': int32
        'was_split': bool
        'filters': set<str>
        'info': struct {
            SB: array<int32>,
            MQRankSum: float64,
            VarDP: int32,
            AS_FS: float64,
            AS_ReadPosRankSum: float64,
            AS_pab_max: float64,
            AS_QD: float32,
            AS_MQ: float64,
            AS_QUALapprox: int64,
            QD: float32,
            AS_MQRankSum: float64,
            FS: float64,
            MQ: float64,
            ReadPosRankSum: float64,
            QUALapprox: int64,
            AS_SB_TABLE: array<int32>,
            AS_VarDP: int32,
            AS_SOR: float64,
            SOR: float64,
            transmitted_singleton: bool,
            omni: bool,
            mills: bool,
            monoallelic: bool,
            InbreedingCoeff: float32,
            AS_VQSLOD: float64
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
        'hgdp_tgp_freq': array<struct {
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
        'gnomad_popmax': struct {
            AC: int32,
            AF: float64,
            AN: int32,
            homozygote_count: int32,
            pop: str,
            faf95: float64
        }
        'gnomad_faf': array<struct {
            faf95: float64,
            faf99: float64
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
        'gnomad_age_hist_het': struct {
            bin_edges: array<float64>,
            bin_freq: array<int64>,
            n_smaller: int64,
            n_larger: int64
        }
        'gnomad_age_hist_hom': struct {
            bin_edges: array<float64>,
            bin_freq: array<int64>,
            n_smaller: int64,
            n_larger: int64
        }
        'cadd': struct {
            phred: float32,
            raw_score: float32,
            has_duplicate: bool
        }
        'revel': struct {
            revel_score: float64,
            has_duplicate: bool
        }
        'splice_ai': struct {
            splice_ai_score: float32,
            splice_consequence: str,
            has_duplicate: bool
        }
        'primate_ai': struct {
            primate_ai_score: float32,
            has_duplicate: bool
        }
    ----------------------------------------
    Entry fields:
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
        'adj': bool
    ----------------------------------------
    Column key: ['s']
    Row key: ['locus', 'alleles']
    ----------------------------------------
