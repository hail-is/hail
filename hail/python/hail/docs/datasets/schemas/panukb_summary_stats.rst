.. _panukb_summary_stats:

panukb_summary_stats
====================

*  **Versions:** 0.3
*  **Reference genome builds:** GRCh37
*  **Type:** :class:`hail.MatrixTable`

Schema (0.3, GRCh37)
~~~~~~~~~~~~~~~~~~~~

.. code-block:: text

    ----------------------------------------
    Global fields:
        None
    ----------------------------------------
    Column fields:
        'trait_type': str
        'phenocode': str
        'pheno_sex': str
        'coding': str
        'modifier': str
        'pheno_data': array<struct {
            n_cases: int32, 
            n_controls: int32, 
            heritability: float64, 
            saige_version: str, 
            inv_normalized: bool, 
            pop: str
        }>
        'description': str
        'description_more': str
        'coding_description': str
        'category': str
        'n_cases_full_cohort_both_sexes': int64
        'n_cases_full_cohort_females': int64
        'n_cases_full_cohort_males': int64
    ----------------------------------------
    Row fields:
        'locus': locus<GRCh37>
        'alleles': array<str>
        'gene': str
        'annotation': str
    ----------------------------------------
    Entry fields:
        'summary_stats': array<struct {
            AF_Allele2: float64, 
            imputationInfo: float64, 
            BETA: float64, 
            SE: float64, 
            `p.value.NA`: float64, 
            `AF.Cases`: float64, 
            `AF.Controls`: float64, 
            Pvalue: float64, 
            low_confidence: bool
        }>
    ----------------------------------------
    Column key: ['trait_type', 'phenocode', 'pheno_sex', 'coding', 'modifier']
    Row key: ['locus', 'alleles']
    ----------------------------------------
