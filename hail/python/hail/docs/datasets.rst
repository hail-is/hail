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
strings to the :func:`.load_dataset` function. The available dataset names,
versions, and reference genome builds are listed in the table below.

===================================== ========== =================
Name                                  Versions   Reference Genomes
===================================== ========== =================
:ref:`1000_genomes`                   phase3     GRCh37, GRCh38   
:ref:`Ensembl_human_reference_genome` release_93 GRCh37, GRCh38   
:ref:`GERP_scores`                    GERP++     GRCh37, GRCh38   
:ref:`GTEx_eQTL_associations`         v7         GRCh37           
:ref:`GTEx_exons`                     v7         GRCh37, GRCh38   
:ref:`GTEx_genes`                     v7         GRCh37, GRCh38   
:ref:`GTEx_transcripts`               v7         GRCh37, GRCh38   
===================================== ========== =================

.. toctree::
    :hidden:

    datasets/1000_genomes.rst
    datasets/Ensembl_human_reference_genome.rst
    datasets/GERP_scores.rst
    datasets/GTEx_eQTL_associations.rst
    datasets/GTEx_exons.rst
    datasets/GTEx_genes.rst
    datasets/GTEx_transcripts.rst
