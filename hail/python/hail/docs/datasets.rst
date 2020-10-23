.. _sec-datasets:

:tocdepth: 1

========
Datasets
========

.. warning::
    All functionality described on this page is experimental.
    Datasets and method are subject to change.

This page describes genetic datasets that are hosted in a public repository
on Google Cloud Platform and are available for use through Hail's
:func:`.load_dataset` function.

To load a dataset from this repository into a Hail pipeline, provide the name,
version, and reference genome build of the dataset you would like to use as
strings to the :func:`.load_dataset` function. You will also need to provide
the region ('us' or 'eu') to access the appropriate bucket. The available
dataset names, versions, and reference genome builds are listed in the table
below.

=================================================== ========== =================
Name                                                Versions   Reference Genomes
=================================================== ========== =================
:ref:`1000_Genomes_autosomes`                       phase_3     GRCh37, GRCh38
:ref:`1000_Genomes_chrMT`                           phase_3     GRCh37
:ref:`1000_Genomes_chrX`                            phase_3     GRCh37, GRCh38
:ref:`1000_Genomes_chrY`                            phase_3     GRCh37, GRCh38
:ref:`CADD`                                         1.4         GRCh37, GRCh38
:ref:`DANN`                                         None        GRCh37, GRCh38
:ref:`Ensembl_homo_sapiens_low_complexity_regions`  release_95  GRCh37, GRCh38
:ref:`Ensembl_homo_sapiens_reference_genome`        release_95  GRCh37, GRCh38
:ref:`GTEx_RNA_seq_gene_read_counts`                v7          GRCh37
:ref:`GTEx_RNA_seq_gene_TPMs`                       v7          GRCh37
:ref:`GTEx_RNA_seq_junction_read_counts`            v7          GRCh37
:ref:`UK_Biobank_Rapid_GWAS_both_sexes`             v2          GRCh37
:ref:`UK_Biobank_Rapid_GWAS_female`                 v2          GRCh37
:ref:`UK_Biobank_Rapid_GWAS_male`                   v2          GRCh37
:ref:`clinvar_gene_summary`                         2019-07     None
:ref:`clinvar_variant_summary`                      2019-07     GRCh37, GRCh38
:ref:`dbNSFP_genes`                                 4.0         None
:ref:`dbNSFP_variants`                              4.0         GRCh37, GRCh38
:ref:`gencode`                                      v19, v31    GRCh37, GRCh38
:ref:`gerp_elements`                                hg19        GRCh37, GRCh38
:ref:`gerp_scores`                                  hg19        GRCh37, GRCh38
:ref:`gnomad_exome_sites`                           2.1.1       GRCh37, GRCh38
:ref:`gnomad_genome_sites`                          2.1.1       GRCh37, GRCh38
:ref:`gnomad_lof_metrics`                           2.1.1       GRCh37, GRCh38
:ref:`ldsc_baselineLD_annotations`                  2.2         GRCh37
:ref:`ldsc_baselineLD_ldscores`                     2.2         GRCh37
:ref:`ldsc_baseline_ldscores`                       1.1         GRCh37
=================================================== ========== =================

.. toctree::
    :hidden:

    datasets/1000_Genomes_autosomes.rst
    datasets/1000_Genomes_chrMT.rst
    datasets/1000_Genomes_chrX.rst
    datasets/1000_Genomes_chrY.rst
    datasets/CADD.rst
    datasets/DANN.rst
    datasets/Ensembl_homo_sapiens_low_complexity_regions.rst
    datasets/Ensembl_homo_sapiens_reference_genome.rst
    datasets/GTEx_RNA_seq_gene_read_counts.rst
    datasets/GTEx_RNA_seq_gene_TPMs.rst
    datasets/GTEx_RNA_seq_junction_read_counts.rst
    datasets/UK_Biobank_Rapid_GWAS_both_sexes.rst
    datasets/UK_Biobank_Rapid_GWAS_female.rst
    datasets/UK_Biobank_Rapid_GWAS_male.rst
    datasets/clinvar_gene_summary.rst
    datasets/clinvar_variant_summary.rst
    datasets/dbNSFP_genes.rst
    datasets/dbNSFP_variants.rst
    datasets/gencode.rst
    datasets/gerp_elements.rst
    datasets/gerp_scores.rst
    datasets/gnomad_exome_sites.rst
    datasets/gnomad_genome_sites.rst
    datasets/gnomad_lof_metrics.rst
    datasets/ldsc_baselineLD_annotations.rst
    datasets/ldsc_baselineLD_ldscores.rst
    datasets/ldsc_baseline_ldscores.rst

.. raw:: html
   :file: _static/datasets/datasets.html