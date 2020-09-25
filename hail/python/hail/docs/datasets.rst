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
dataset names, versions, and reference genome builds are listed in the table below.

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
