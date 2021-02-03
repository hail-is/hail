.. _clinvar_gene_summary:

clinvar_gene_summary
====================

*  **Versions:** 2019-07
*  **Reference genome builds:** None
*  **Type:** :class:`hail.Table`

Schema (2019-07, None)
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: text

    ----------------------------------------
    Global fields:
        None
    ----------------------------------------
    Row fields:
        'GeneID': int32
        'Total_submissions': int32
        'Total_alleles': int32
        'Submissions_reporting_this_gene': int32
        'Alleles_reported_Pathogenic_Likely_pathogenic': int32
        'Gene_MIM_number': int32
        'Number_uncertain': int32
        'Number_with_conflicts': int32
        'gene_name': str
    ----------------------------------------
    Key: ['gene_name']
    ----------------------------------------

