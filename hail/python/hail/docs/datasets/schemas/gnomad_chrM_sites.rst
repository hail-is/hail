.. _gnomad_chrM_sites:

gnomad_chrM_sites
=================

*  **Versions:** 3.1
*  **Reference genome builds:** GRCh38
*  **Type:** :class:`hail.Table`

Schema (3.1, GRCh38)
~~~~~~~~~~~~~~~~~~~~

.. code-block:: text

    ----------------------------------------
    Global fields:
        'vep_version': str
        'dbsnp_version': str
        'hap_order': array<str>
        'dp_hist_all_variants_bin_freq': array<int32>
        'dp_hist_all_variants_n_larger': int32
        'dp_hist_all_variants_bin_edges': array<float64>
        'mq_hist_all_variants_bin_freq': array<int32>
        'mq_hist_all_variants_n_larger': int32
        'mq_hist_all_variants_bin_edges': array<float64>
        'tlod_hist_all_variants_bin_freq': array<int32>
        'tlod_hist_all_variants_n_larger': int32
        'tlod_hist_all_variants_bin_edges': array<float64>
        'age_hist_all_samples_bin_freq': array<int32>
        'age_hist_all_samples_n_larger': int32
        'age_hist_all_samples_n_smaller': int32
        'age_hist_all_samples_bin_edges': array<float64>
    ----------------------------------------
    Row fields:
        'locus': locus<GRCh38>
        'alleles': array<str>
        'rsid': str
        'qual': float64
        'filters': set<str>
        'variant_collapsed': str
        'hap_defining_variant': bool
        'pon_mt_trna_prediction': str
        'pon_ml_probability_of_pathogenicity': float64
        'mitotip_score': float64
        'mitotip_trna_prediction': str
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
        'common_low_heteroplasmy': bool
        'base_qual_hist': array<int64>
        'position_hist': array<int64>
        'strand_bias_hist': array<int64>
        'weak_evidence_hist': array<int64>
        'contamination_hist': array<int64>
        'heteroplasmy_below_10_percent_hist': array<int64>
        'excluded_AC': int64
        'AN': int64
        'AC_hom': int64
        'AC_het': int64
        'hl_hist': struct {
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
        'dp_hist_alt': struct {
            bin_edges: array<float64>,
            bin_freq: array<int64>,
            n_smaller: int64,
            n_larger: int64
        }
        'dp_mean': float64
        'mq_mean': float64
        'tlod_mean': float64
        'AF_hom': float32
        'AF_het': float32
        'max_hl': float64
        'hap_AN': array<int64>
        'hap_AC_het': array<int64>
        'hap_AC_hom': array<int64>
        'hap_AF_hom': array<float32>
        'hap_AF_het': array<float32>
        'hap_hl_hist': array<array<int64>>
        'hap_faf_hom': array<float64>
        'hapmax_AF_hom': str
        'hapmax_AF_het': str
        'faf_hapmax_hom': float64
        'age_hist_hom': struct {
            bin_edges: array<float64>,
            bin_freq: array<int64>,
            n_smaller: int64,
            n_larger: int64
        }
        'age_hist_het': struct {
            bin_edges: array<float64>,
            bin_freq: array<int64>,
            n_smaller: int64,
            n_larger: int64
        }
    ----------------------------------------
    Key: ['locus', 'alleles']
    ----------------------------------------

