.. _gnomad_genome_sites:

gnomad_genome_sites
===================

*  **Versions:** 2.1.1, 3.1, 3.1.1, 3.1.2
*  **Reference genome builds:** GRCh37, GRCh38
*  **Type:** :class:`hail.Table`

Schema (3.1.2, GRCh38)
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: text

    ----------------------------------------
    Global fields:
        'freq_meta': array<dict<str, str>>
        'freq_index_dict': dict<str, int32>
        'faf_index_dict': dict<str, int32>
        'faf_meta': array<dict<str, str>>
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
            model_id: str,
            snv_training_variables: array<str>,
            indel_training_variables: array<str>
        }
        'age_distribution': struct {
            bin_edges: array<float64>,
            bin_freq: array<int32>,
            n_smaller: int32,
            n_larger: int32
        }
        'freq_sample_count': array<int32>
    ----------------------------------------
    Row fields:
        'locus': locus<GRCh38>
        'alleles': array<str>
        'freq': array<struct {
            AC: int32,
            AF: float64,
            AN: int32,
            homozygote_count: int32
        }>
        'raw_qual_hists': struct {
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
        'popmax': struct {
            AC: int32,
            AF: float64,
            AN: int32,
            homozygote_count: int32,
            pop: str,
            faf95: float64
        }
        'qual_hists': struct {
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
        'faf': array<struct {
            faf95: float64,
            faf99: float64
        }>
        'a_index': int32
        'was_split': bool
        'rsid': set<str>
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
            singleton: bool,
            transmitted_singleton: bool,
            omni: bool,
            mills: bool,
            monoallelic: bool,
            AS_VQSLOD: float64,
            InbreedingCoeff: float64
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
        'age_hist_het': struct {
            bin_edges: array<float64>,
            bin_freq: array<int64>,
            n_smaller: int64,
            n_larger: int64
        }
        'age_hist_hom': struct {
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
    Key: ['locus', 'alleles']
    ----------------------------------------
