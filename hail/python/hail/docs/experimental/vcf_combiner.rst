VCF Combiner
============

Library functions for combining gVCFS and sparse matrix tables into
larger sparse matrix tables.

What this module provides:
    - A Sensible way to transform input gVCFS.
    - The combining function.

What this module does not provide:
    - Any way to repartition the data.

There are two additional functions of note for working with sparse data in the
main experimental module :func:`.sparse_split_multi`, which splits multi-alleleics
in a sparse matrix table, and :func:`.densify` which converts a sparse matrix table
to a dense one.

.. currentmodule:: hail.experimental.vcf_combiner

.. autosummary::

    combine_gvcfs
    transform_one
    lgt_to_gt

.. autofunction:: combine_gvcfs
.. autofunction:: transform_one
.. autofunction:: lgt_to_gt
