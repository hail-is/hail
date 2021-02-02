.. _clinvar_variant_summary:

clinvar_variant_summary
=======================

*  **Versions:** 2019-07
*  **Reference genome builds:** GRCh37, GRCh38
*  **Type:** :class:`hail.Table`

Schema (2019-07, GRCh37)
~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: text

    ----------------------------------------
    Global fields:
        None
    ----------------------------------------
    Row fields:
        'Type': str
        'Name': str
        'GeneID': int32
        'GeneSymbol': str
        'HGNC_ID': str
        'ClinicalSignificance': str
        'ClinSigSimple': int32
        'LastEvaluated': str
        'RS# (dbSNP)': int32
        'nsv/esv (dbVar)': str
        'RCVaccession': str
        'PhenotypeIDS': str
        'PhenotypeList': str
        'Origin': str
        'OriginSimple': str
        'Assembly': str
        'ChromosomeAccession': str
        'ReferenceAllele': str
        'AlternateAllele': str
        'Cytogenetic': str
        'ReviewStatus': str
        'NumberSubmitters': int32
        'Guidelines': str
        'TestedInGTR': str
        'OtherIDs': str
        'SubmitterCategories': int32
        'VariationID': int32
        'interval': interval<locus<GRCh37>>
        'AlleleID': int32
    ----------------------------------------
    Key: ['interval']
    ----------------------------------------

