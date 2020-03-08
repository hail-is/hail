`ldscsim`
=========

Models for SNP effects:
    - Infinitesimal (can simulate n correlated traits)
    - Spike & slab (can simulate 2 correlated traits)
    - Annotation-informed

Features:
   - Field aggregation tools for annotation-informed model and
     population stratification with many covariates.
   - Automatic adjustment of genetic correlation parameters
     to allow for the joint simulation of up to 100 randomly
     correlated phenotypes.
   - Methods for binarizing phenotypes to have a certain prevalence
     and for adding ascertainment bias to binarized phenotypes.


.. currentmodule:: hail.experimental.ldscsim

.. autosummary::

    simulate_phenotypes
    make_betas
    multitrait_inf
    multitrait_ss
    get_cov_matrix
    calculate_phenotypes
    normalize_genotypes
    annotate_all
    ascertainment_bias
    binarize
    agg_fields
    get_coef_dict


.. autofunction:: simulate_phenotypes
.. autofunction:: make_betas
.. autofunction:: multitrait_inf
.. autofunction:: multitrait_ss
.. autofunction:: get_cov_matrix
.. autofunction:: calculate_phenotypes
.. autofunction:: normalize_genotypes
.. autofunction:: annotate_all
.. autofunction:: ascertainment_bias
.. autofunction:: binarize
.. autofunction:: agg_fields
.. autofunction:: get_coef_dict
