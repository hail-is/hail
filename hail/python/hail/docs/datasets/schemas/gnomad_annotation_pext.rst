.. _gnomad_annotation_pext:

gnomad_annotation_pext
======================

*  **Versions:** 2.1.1
*  **Reference genome builds:** GRCh37
*  **Type:** :class:`hail.Table`

Schema (2.1.1, GRCh37)
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: text

    ----------------------------------------
    Global fields:
        None
    ----------------------------------------
    Row fields:
        'context': str
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
        'locus': locus<GRCh37>
        'alleles': array<str>
        'a_index': int32
        'was_split': bool
        'methylation': struct {
            NUM: int32,
            MEAN: float64,
            GTE50: int32,
            GTE60: int32,
            GTE70: int32,
            GTE80: int32,
            GTE90: int32,
            GTE100: int32
        }
        'coverage': struct {
            exomes: struct {
                row_id: int64,
                mean: float64,
                median: int32,
                over_1: float64,
                over_5: float64,
                over_10: float64,
                over_15: float64,
                over_20: float64,
                over_25: float64,
                over_30: float64,
                over_50: float64,
                over_100: float64
            },
            genomes: struct {
                row_id: int64,
                mean: float64,
                median: int32,
                over_1: float64,
                over_5: float64,
                over_10: float64,
                over_15: float64,
                over_20: float64,
                over_25: float64,
                over_30: float64,
                over_50: float64,
                over_100: float64
            }
        }
        'gerp': float64
        'tx_annotation': array<struct {
            ensg: str,
            csq: str,
            symbol: str,
            lof: str,
            lof_flag: str,
            Cells_Transformedfibroblasts: float64,
            Prostate: float64,
            Spleen: float64,
            Brain_FrontalCortex_BA9_: float64,
            SmallIntestine_TerminalIleum: float64,
            MinorSalivaryGland: float64,
            Artery_Coronary: float64,
            Skin_SunExposed_Lowerleg_: float64,
            Cells_EBV_transformedlymphocytes: float64,
            Brain_Hippocampus: float64,
            Esophagus_Muscularis: float64,
            Brain_Nucleusaccumbens_basalganglia_: float64,
            Artery_Tibial: float64,
            Brain_Hypothalamus: float64,
            Adipose_Visceral_Omentum_: float64,
            Cervix_Ectocervix: float64,
            Brain_Spinalcord_cervicalc_1_: float64,
            Brain_CerebellarHemisphere: float64,
            Nerve_Tibial: float64,
            Breast_MammaryTissue: float64,
            Liver: float64,
            Skin_NotSunExposed_Suprapubic_: float64,
            AdrenalGland: float64,
            Vagina: float64,
            Pancreas: float64,
            Lung: float64,
            FallopianTube: float64,
            Pituitary: float64,
            Muscle_Skeletal: float64,
            Colon_Transverse: float64,
            Artery_Aorta: float64,
            Heart_AtrialAppendage: float64,
            Adipose_Subcutaneous: float64,
            Esophagus_Mucosa: float64,
            Heart_LeftVentricle: float64,
            Brain_Cerebellum: float64,
            Brain_Cortex: float64,
            Thyroid: float64,
            Brain_Substantianigra: float64,
            Kidney_Cortex: float64,
            Uterus: float64,
            Stomach: float64,
            WholeBlood: float64,
            Bladder: float64,
            Brain_Anteriorcingulatecortex_BA24_: float64,
            Brain_Putamen_basalganglia_: float64,
            Brain_Caudate_basalganglia_: float64,
            Colon_Sigmoid: float64,
            Cervix_Endocervix: float64,
            Ovary: float64,
            Esophagus_GastroesophagealJunction: float64,
            Testis: float64,
            Brain_Amygdala: float64,
            mean_proportion: float64
        }>
    ----------------------------------------
    Key: ['locus', 'alleles']
    ----------------------------------------
