.. _gnomad_hgdp_1kg_subset_variant_annotations:

gnomad_hgdp_1kg_subset_variant_annotations
==========================================

*  **Versions:** 3.1.2
*  **Reference genome builds:** GRCh38
*  **Type:** :class:`hail.Table`

Schema (3.1.2, GRCh38)
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: text

    ----------------------------------------
    Global fields:
        'global_annotation_descriptions': struct {
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
    Row fields:
        'locus': locus<GRCh38>
        'alleles': array<str>
        'a_index': int32
        'was_split': bool
        'rsid': set<str>
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
        'AS_lowqual': bool
        'telomere_or_centromere': bool
    ----------------------------------------
    Key: ['locus', 'alleles']
    ----------------------------------------
