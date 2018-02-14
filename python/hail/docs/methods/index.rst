Methods
=======

.. currentmodule:: hail.methods

.. toctree::
    :maxdepth: 2

    import
    export
    regression
    statgen
    misc

.. rubric:: Import

.. autosummary::

    import_bed
    import_bgen
    import_fam
    import_gen
    import_interval_list
    import_plink
    import_vcf
    read_matrix_table
    read_table


.. rubric:: Export

.. autosummary::

    export_cassandra
    export_gen
    export_plink
    export_solr
    export_vcf


.. rubric:: Regression

.. autosummary::

    linreg
    logreg
    lmmreg


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
    SplitMulti
    split_multi_hts
    tdt
    trio_matrix
    variant_qc
    vep


.. rubric:: Miscellaneous

.. autosummary::

    get_vcf_metadata
    maximal_independent_set
    pca
    rename_duplicates
    sample_rows
    skat
