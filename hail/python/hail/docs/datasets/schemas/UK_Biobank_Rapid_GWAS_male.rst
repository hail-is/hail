.. _UK_Biobank_Rapid_GWAS_male:

UK_Biobank_Rapid_GWAS_male
==========================

*  **Versions:** v2
*  **Reference genome builds:** GRCh37
*  **Type:** :class:`hail.MatrixTable`

Schema (v2, GRCh37)
~~~~~~~~~~~~~~~~~~~

.. code-block:: text

    ----------------------------------------
    Global fields:
        'metadata': struct {
            name: str,
            version: str,
            reference_genome: str,
            n_rows: int32,
            n_cols: int32,
            n_partitions: int32
        }
    ----------------------------------------
    Column fields:
        'phenotype': str
        'description': str
        'variable_type': str
        'source': str
        'n_non_missing': int32
        'n_missing': int32
        'n_controls': int32
        'n_cases': int32
        'PHESANT_transformation': str
        'notes': str
    ----------------------------------------
    Row fields:
        'locus': locus<GRCh37>
        'alleles': array<str>
        'variant': str
        'minor_allele': str
        'minor_AF': float64
        'rsid': str
        'varid': str
        'consequence': str
        'consequence_category': str
        'info': float64
        'call_rate': float64
        'alt_AC': int32
        'AF': float64
        'p_hwe': float64
        'n_called': int32
        'n_not_called': int32
        'n_hom_ref': int32
        'n_het': int32
        'n_hom_var': int32
        'n_non_ref': int32
        'r_heterozygosity': float64
        'r_het_hom_var': float64
        'r_expected_het_frequency': float64
    ----------------------------------------
    Entry fields:
        'expected_case_minor_AC': float64
        'expected_min_category_minor_AC': float64
        'low_confidence_variant': bool
        'n_complete_samples': int32
        'AC': float64
        'ytx': float64
        'beta': float64
        'se': float64
        'tstat': float64
        'pval': float64
    ----------------------------------------
    Column key: ['phenotype']
    Row key: ['locus', 'alleles']
    ----------------------------------------

