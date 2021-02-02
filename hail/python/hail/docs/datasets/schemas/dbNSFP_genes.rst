.. _dbNSFP_genes:

dbNSFP_genes
============

*  **Versions:** 4.0
*  **Reference genome builds:** None
*  **Type:** :class:`hail.Table`

Schema (4.0, None)
~~~~~~~~~~~~~~~~~~

.. code-block:: text

    ----------------------------------------
    Global fields:
        None
    ----------------------------------------
    Row fields:
        'Gene_name': str
        'Ensembl_gene': str
        'chr': str
        'Gene_old_names': str
        'Gene_other_names': str
        'Uniprot_acc(HGNC/Uniprot)': str
        'Uniprot_id(HGNC/Uniprot)': str
        'Entrez_gene_id': int32
        'CCDS_id': str
        'Refseq_id': str
        'ucsc_id': str
        'MIM_id': str
        'OMIM_id': int32
        'Gene_full_name': str
        'Pathway(Uniprot)': str
        'Pathway(BioCarta)_short': str
        'Pathway(BioCarta)_full': str
        'Pathway(ConsensusPathDB)': str
        'Pathway(KEGG)_id': str
        'Pathway(KEGG)_full': str
        'Function_description': str
        'Disease_description': str
        'MIM_phenotype_id': str
        'MIM_disease': str
        'Orphanet_disorder_id': str
        'Orphanet_disorder': str
        'Orphanet_association_type': str
        'Trait_association(GWAS)': str
        'GO_biological_process': str
        'GO_cellular_component': str
        'GO_molecular_function': str
        'Tissue_specificity(Uniprot)': str
        'Expression(egenetics)': str
        'Expression(GNF/Atlas)': str
        'Interactions(IntAct)': str
        'Interactions(BioGRID)': str
        'Interactions(ConsensusPathDB)': str
        'P(HI)': float64
        'HIPred_score': float64
        'HIPred': str
        'GHIS': float64
        'P(rec)': float64
        'Known_rec_info': str
        'RVIS_EVS': float64
        'RVIS_percentile_EVS': float64
        'LoF-FDR_ExAC': float64
        'RVIS_ExAC': float64
        'RVIS_percentile_ExAC': float64
        'ExAC_pLI': float64
        'ExAC_pRec': float64
        'ExAC_pNull': float64
        'ExAC_nonTCGA_pLI': float64
        'ExAC_nonTCGA_pRec': float64
        'ExAC_nonTCGA_pNull': float64
        'ExAC_nonpsych_pLI': float64
        'ExAC_nonpsych_pRec': float64
        'ExAC_nonpsych_pNull': float64
        'gnomAD_pLI': str
        'gnomAD_pRec': str
        'gnomAD_pNull': str
        'ExAC_del.score': float64
        'ExAC_dup.score': float64
        'ExAC_cnv.score': float64
        'ExAC_cnv_flag': str
        'GDI': float64
        'GDI-Phred': float64
        'Gene damage prediction (all disease-causing genes)': str
        'Gene damage prediction (all Mendelian disease-causing genes)': str
        'Gene damage prediction (Mendelian AD disease-causing genes)': str
        'Gene damage prediction (Mendelian AR disease-causing genes)': str
        'Gene damage prediction (all PID disease-causing genes)': str
        'Gene damage prediction (PID AD disease-causing genes)': str
        'Gene damage prediction (PID AR disease-causing genes)': str
        'Gene damage prediction (all cancer disease-causing genes)': str
        'Gene damage prediction (cancer recessive disease-causing genes)': str
        'Gene damage prediction (cancer dominant disease-causing genes)': str
        'LoFtool_score': float64
        'SORVA_LOF_MAF0.005_HetOrHom': float64
        'SORVA_LOF_MAF0.005_HomOrCompoundHet': float64
        'SORVA_LOF_MAF0.001_HetOrHom': float64
        'SORVA_LOF_MAF0.001_HomOrCompoundHet': float64
        'SORVA_LOForMissense_MAF0.005_HetOrHom': float64
        'SORVA_LOForMissense_MAF0.005_HomOrCompoundHet': float64
        'SORVA_LOForMissense_MAF0.001_HetOrHom': float64
        'SORVA_LOForMissense_MAF0.001_HomOrCompoundHet': float64
        'Essential_gene': str
        'Essential_gene_CRISPR': str
        'Essential_gene_CRISPR2': str
        'Essential_gene_gene-trap': str
        'Gene_indispensability_score': float64
        'Gene_indispensability_pred': str
        'MGI_mouse_gene': str
        'MGI_mouse_phenotype': str
        'ZFIN_zebrafish_gene': str
        'ZFIN_zebrafish_structure': str
        'ZFIN_zebrafish_phenotype_quality': str
        'ZFIN_zebrafish_phenotype_tag': str
    ----------------------------------------
    Key: ['Gene_name']
    ----------------------------------------

