.. _ldsc_baselineLD_annotations:

ldsc_baselineLD_annotations
===========================

*  **Versions:** 2.2
*  **Reference genome builds:** GRCh37
*  **Type:** :class:`hail.Table`

Schema (2.2, GRCh37)
~~~~~~~~~~~~~~~~~~~~

.. code-block:: text

    ----------------------------------------
    Global fields:
        'metadata': struct {
            name: str,
            reference_genome: str,
            n_rows: int32,
            n_partitions: int32
        }
    ----------------------------------------
    Row fields:
        'SNP': str
        'baseL2': float64
        'Coding_UCSCL2': float64
        'Coding_UCSC.flanking.500L2': float64
        'Conserved_LindbladTohL2': float64
        'Conserved_LindbladToh.flanking.500L2': float64
        'CTCF_HoffmanL2': float64
        'CTCF_Hoffman.flanking.500L2': float64
        'DGF_ENCODEL2': float64
        'DGF_ENCODE.flanking.500L2': float64
        'DHS_peaks_TrynkaL2': float64
        'DHS_TrynkaL2': float64
        'DHS_Trynka.flanking.500L2': float64
        'Enhancer_AnderssonL2': float64
        'Enhancer_Andersson.flanking.500L2': float64
        'Enhancer_HoffmanL2': float64
        'Enhancer_Hoffman.flanking.500L2': float64
        'FetalDHS_TrynkaL2': float64
        'FetalDHS_Trynka.flanking.500L2': float64
        'H3K27ac_HniszL2': float64
        'H3K27ac_Hnisz.flanking.500L2': float64
        'H3K27ac_PGC2L2': float64
        'H3K27ac_PGC2.flanking.500L2': float64
        'H3K4me1_peaks_TrynkaL2': float64
        'H3K4me1_TrynkaL2': float64
        'H3K4me1_Trynka.flanking.500L2': float64
        'H3K4me3_peaks_TrynkaL2': float64
        'H3K4me3_TrynkaL2': float64
        'H3K4me3_Trynka.flanking.500L2': float64
        'H3K9ac_peaks_TrynkaL2': float64
        'H3K9ac_TrynkaL2': float64
        'H3K9ac_Trynka.flanking.500L2': float64
        'Intron_UCSCL2': float64
        'Intron_UCSC.flanking.500L2': float64
        'PromoterFlanking_HoffmanL2': float64
        'PromoterFlanking_Hoffman.flanking.500L2': float64
        'Promoter_UCSCL2': float64
        'Promoter_UCSC.flanking.500L2': float64
        'Repressed_HoffmanL2': float64
        'Repressed_Hoffman.flanking.500L2': float64
        'SuperEnhancer_HniszL2': float64
        'SuperEnhancer_Hnisz.flanking.500L2': float64
        'TFBS_ENCODEL2': float64
        'TFBS_ENCODE.flanking.500L2': float64
        'Transcr_HoffmanL2': float64
        'Transcr_Hoffman.flanking.500L2': float64
        'TSS_HoffmanL2': float64
        'TSS_Hoffman.flanking.500L2': float64
        'UTR_3_UCSCL2': float64
        'UTR_3_UCSC.flanking.500L2': float64
        'UTR_5_UCSCL2': float64
        'UTR_5_UCSC.flanking.500L2': float64
        'WeakEnhancer_HoffmanL2': float64
        'WeakEnhancer_Hoffman.flanking.500L2': float64
        'GERP.NSL2': float64
        'GERP.RSsup4L2': float64
        'MAFbin1L2': float64
        'MAFbin2L2': float64
        'MAFbin3L2': float64
        'MAFbin4L2': float64
        'MAFbin5L2': float64
        'MAFbin6L2': float64
        'MAFbin7L2': float64
        'MAFbin8L2': float64
        'MAFbin9L2': float64
        'MAFbin10L2': float64
        'MAF_Adj_Predicted_Allele_AgeL2': float64
        'MAF_Adj_LLD_AFRL2': float64
        'Recomb_Rate_10kbL2': float64
        'Nucleotide_Diversity_10kbL2': float64
        'Backgrd_Selection_StatL2': float64
        'CpG_Content_50kbL2': float64
        'MAF_Adj_ASMCL2': float64
        'GTEx_eQTL_MaxCPPL2': float64
        'BLUEPRINT_H3K27acQTL_MaxCPPL2': float64
        'BLUEPRINT_H3K4me1QTL_MaxCPPL2': float64
        'BLUEPRINT_DNA_methylation_MaxCPPL2': float64
        'synonymousL2': float64
        'non_synonymousL2': float64
        'Conserved_Vertebrate_phastCons46wayL2': float64
        'Conserved_Vertebrate_phastCons46way.flanking.500L2': float64
        'Conserved_Mammal_phastCons46wayL2': float64
        'Conserved_Mammal_phastCons46way.flanking.500L2': float64
        'Conserved_Primate_phastCons46wayL2': float64
        'Conserved_Primate_phastCons46way.flanking.500L2': float64
        'BivFlnkL2': float64
        'BivFlnk.flanking.500L2': float64
        'Human_Promoter_VillarL2': float64
        'Human_Promoter_Villar.flanking.500L2': float64
        'Human_Enhancer_VillarL2': float64
        'Human_Enhancer_Villar.flanking.500L2': float64
        'Ancient_Sequence_Age_Human_PromoterL2': float64
        'Ancient_Sequence_Age_Human_Promoter.flanking.500L2': float64
        'Ancient_Sequence_Age_Human_EnhancerL2': float64
        'Ancient_Sequence_Age_Human_Enhancer.flanking.500L2': float64
        'Human_Enhancer_Villar_Species_Enhancer_CountL2': float64
        'Human_Promoter_Villar_ExACL2': float64
        'Human_Promoter_Villar_ExAC.flanking.500L2': float64
        'locus': locus<GRCh37>
    ----------------------------------------
    Key: ['locus']
    ----------------------------------------

