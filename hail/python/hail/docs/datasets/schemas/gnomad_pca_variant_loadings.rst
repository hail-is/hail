.. _gnomad_pca_variant_loadings:

gnomad_pca_variant_loadings
===========================

*  **Versions:** 2.1, 3.1
*  **Reference genome builds:** GRCh37, GRCh38
*  **Type:** :class:`hail.Table`

Schema (3.1, GRCh38)
~~~~~~~~~~~~~~~~~~~~

.. code-block:: text

    ----------------------------------------
    Global fields:
        None
    ----------------------------------------
    Row fields:
        'locus': locus<GRCh38>
        'alleles': array<str>
        'loadings': array<float64>
        'pca_af': float64
    ----------------------------------------
    Key: ['locus', 'alleles']
    ----------------------------------------
