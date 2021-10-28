.. _sec-vds:

Variant Dataset
===============

The Variant Dataset is a set of Python classes and functions for efficiently working with large
sequencing datasets. Variant Datasets store reference and variant data in two distinct MatrixTables.

.. warning::

   Hail 0.1 also had a Variant Dataset class. Although their interfaces are similar, they should not
   be considered interchangeable.

.. currentmodule:: hail.vds

.. rubric:: Variant Dataset

.. autosummary::
    :nosignatures:
    :toctree: ./
    :template: class.rst

    VariantDataset
    read_vds
    lgt_to_gt
    filter_intervals
    filter_samples
    filter_variants
    sample_qc
    split_multi
    to_dense_mt
    to_merged_sparse_mt
    lgt_to_gt
    write_variant_datasets

.. rubric:: Variant Dataset Combiner

.. currentmodule:: hail.vds.combiner

.. autosummary::
    :nosignatures:
    :toctree: ./
    :template: class.rst

    combine_variant_datasets
    transform_gvcf
    new_combiner
    load_combiner
