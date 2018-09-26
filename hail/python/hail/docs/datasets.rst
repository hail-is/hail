.. _sec-datasets:

:tocdepth: 1

.. warning::
    All functionality described on this page is experimental.
    Datasets and method are subject to change.

This page describes genetic datasets that have already been imported into
Hail-friendly formats and are available for use through Hail's
:meth:`.load_dataset` function.

========
Datasets
========

========================================================= ========================== =====================
Dataset name                                              Reference builds available Type
========================================================= ========================== =====================
:ref:`1000_genomes_phase3_autosomes`                             GRCh37, GRCh38             :class:`.MatrixTable`
:ref:`1000_genomes_phase3_chrMT`                                 GRCh37, GRCh38             :class:`.MatrixTable`
:ref:`1000_genomes_phase3_chrX`                                  GRCh37, GRCh38             :class:`.MatrixTable`
:ref:`1000_genomes_phase3_chrY`                                  GRCh37, GRCh38             :class:`.MatrixTable`
:ref:`dbsnp_build151`                                            GRCh37, GRCh38             :class:`.Table`
:ref:`gerp_scores`                                               GRCh37, GRCh38             :class:`.Table`
:ref:`gtex_v7_eqtl_associations`                                 GRCh37, GRCh38             :class:`.MatrixTable`
:ref:`gtex_v7_eqtl_egenes`                                       GRCh37, GRCh38             :class:`.MatrixTable`
:ref:`gtex_v7_eqtl_normalized_expression_and_covariates`         GRCh37, GRCh38             :class:`.MatrixTable`
:ref:`gtex_v7_eqtl_significant_associations`                     GRCh37, GRCh38             :class:`.MatrixTable`
:ref:`gtex_v7_exon_read_counts`                                  GRCh37, GRCh38             :class:`.MatrixTable`
:ref:`gtex_v7_gene_read_counts`                                  GRCh37, GRCh38             :class:`.MatrixTable`
:ref:`gtex_v7_gene_tpm`                                          GRCh37, GRCh38             :class:`.MatrixTable`
:ref:`gtex_v7_junction_read_counts`                              GRCh37, GRCh38             :class:`.MatrixTable`
:ref:`gtex_v7_transcript_read_counts`                            GRCh37, GRCh38             :class:`.MatrixTable`
:ref:`gtex_v7_transcript_tpm`                                    GRCh37, GRCh38             :class:`.MatrixTable`
:ref:`ukbb_imputed_v3_gwas_results_both_sexes`                   GRCh37, GRCh38             :class:`.MatrixTable`
:ref:`ukbb_imputed_v3_gwas_results_female`                       GRCh37, GRCh38             :class:`.MatrixTable`
:ref:`ukbb_imputed_v3_gwas_results_male`                         GRCh37, GRCh38             :class:`.MatrixTable`
========================================================= ========================== =====================

.. toctree::
    :glob:
    :hidden:

    datasets/*
