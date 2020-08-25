Methods
=======

.. currentmodule:: hail.methods

.. toctree::
    :maxdepth: 2

    impex
    stats
    genetics
    relatedness
    misc

.. rubric:: Import / Export

.. autosummary::

    export_elasticsearch
    export_gen
    export_bgen
    export_plink
    export_vcf
    get_vcf_metadata
    import_bed
    import_bgen
    import_fam
    import_gen
    import_locus_intervals
    import_matrix_table
    import_plink
    import_table
    import_vcf
    import_gvcfs
    index_bgen
    read_matrix_table
    read_table


.. rubric:: Statistics

.. autosummary::

    linear_mixed_model
    linear_mixed_regression_rows
    linear_regression_rows
    logistic_regression_rows
    poisson_regression_rows
    pca
    row_correlation


.. rubric:: Genetics

.. autosummary::

    balding_nichols_model
    concordance
    filter_intervals
    filter_alleles
    filter_alleles_hts
    genetic_relatedness_matrix
    hwe_normalized_pca
    impute_sex
    ld_matrix
    ld_prune
    mendel_errors
    de_novo
    nirvana
    realized_relationship_matrix
    sample_qc
    skat
    lambda_gc
    split_multi
    split_multi_hts
    transmission_disequilibrium_test
    trio_matrix
    variant_qc
    vep


.. rubric:: Relatedness

Hail provides three methods for the inference of relatedness: PLINK-style
identity by descent, KING, and PC-Relate.

- :func:`.identity_by_descent` is appropriate for datasets containing one
  homogeneous population.
- :func:`.king` is appropriate for datasets containing multiple homogeneous
  populations and no admixture. It is also used to prune close relatives before
  using :func:`.pc_relate`.
- :func:`.pc_relate` is appropriate for datasets containing multiple homogeneous
  populations and admixture.

.. autosummary::

    identity_by_descent
    pc_relate
    king

.. rubric:: Miscellaneous

.. autosummary::

    grep
    maximal_independent_set
    rename_duplicates
