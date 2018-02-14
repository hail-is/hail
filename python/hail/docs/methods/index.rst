Methods
=======

.. currentmodule:: hail.methods

.. toctree::
    :maxdepth: 2

    impex
    stats
    statgen
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
    import_interval_list
    import_plink
    import_vcf
    read_matrix_table
    read_table


.. rubric:: Statistics

.. autosummary::

    linreg
    logreg
    lmmreg
    pca


.. rubric:: Statistical Genetics

.. autosummary::

    balding_nichols_model
    concordance
    FilterAlleles
    grm
    hwe_normalized_pca
    ibd
    mendel_errors
    min_rep
    nirvana
    pc_relate
    rrm
    sample_qc
    skat
    SplitMulti
    split_multi_hts
    tdt
    trio_matrix
    variant_qc
    vep


.. rubric:: Miscellaneous

.. autosummary::

    maximal_independent_set
    rename_duplicates
    sample_rows
