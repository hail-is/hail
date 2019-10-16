`vcf_combiner`
==============

Library functions for combining gVCFS and sparse matrix tables into
larger sparse matrix tables.

What this module provides:
    - Sensible ways to transform input gVCFS.
    - Combining functions

What this module does not provide:
    - Any way to repartition the data.

.. currentmodule:: hail.experimental.vcf_combiner

.. autosummary::

    combine_gvcfs
    transform_one
    lgt_to_gt

.. autofunction:: combine_gvcfs
.. autofunction:: transform_one
.. autofunction:: lgt_to_gt
