Methods
=======

.. currentmodule:: hail.methods

.. toctree::
    :maxdepth: 2

    impex
    stats
    genetics
    misc

.. rubric:: Import / Export

.. autosummary::

    export_cassandra
    export_gen
    export_plink
    export_solr
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
    read_matrix_table
    read_table


.. rubric:: Statistics

.. autosummary::

    linear_regression
    logistic_regression
    linear_mixed_regression
    pca


.. rubric:: Genetics

.. autosummary::

    balding_nichols_model
    concordance
    filter_intervals
    FilterAlleles
    genetic_relatedness_matrix
    hwe_normalized_pca
    identity_by_descent
    impute_sex
    ld_prune
    mendel_errors
    min_rep
    nirvana
    pc_relate
    realized_relationship_matrix
    sample_qc
    skat
    split_multi_hts
    SplitMulti
    transmission_disequilibrium_test
    trio_matrix
    variant_qc
    vep


.. rubric:: Miscellaneous

.. autosummary::

    grep
    maximal_independent_set
    rename_duplicates
