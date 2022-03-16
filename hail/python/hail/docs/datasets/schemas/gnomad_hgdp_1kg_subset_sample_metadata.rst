.. _gnomad_hgdp_1kg_subset_sample_metadata:

gnomad_hgdp_1kg_subset_sample_metadata
======================================

*  **Versions:** 3.1.2
*  **Reference genome builds:** GRCh38
*  **Type:** :class:`hail.Table`

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
    ----------------------------------------
    Row fields:
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
    Key: ['s']
    ----------------------------------------
