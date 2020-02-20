VCF Combiner
============

Library functions for combining GVCFS and sparse matrix tables into
larger sparse matrix tables.

What this module provides:
    - A Sensible way to transform input GVCFS.
    - The combining function.

What this module does not provide:
    - Any way to repartition the data.

There are three additional functions of note for working with sparse data in the main experimental
module :func:`.sparse_split_multi`, which splits multi-alleleics in a sparse matrix table,
:func:`.lgt_to_gt`, which converts from local alleles and genotypes to true(global) genotypes, and
:func:`.densify`, which converts a sparse matrix table to a dense one.

.. currentmodule:: hail.experimental.vcf_combiner

.. autosummary::

    combine_gvcfs
    transform_gvcf

Sparse Matrix Tables
--------------------

Sparse matrix tables are a new method of representing VCF style data in a space efficient way. They
are produced them using :func:`transform_gvcf` on an imported GVCF, or by using
:func:`combine_gvcfs` on smaller sparse matrix tables. They have two components that differentiate
them from matrix tables produced by importing VCFs.

* `Sample Level Reference Blocks`_
* `Local Alleles`_

Sample Level Reference Blocks
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

GVCFs represent blocks of homozygous reference calls of similar qualities using one record. For
example: ::

    #CHROM  POS    ID  REF  ALT  INFO       FORMAT    S01
    chr1    14523  .   C    .    END=15000  GT:DP:GQ  0/0:19:40

This record indicates that S01 is homozygous reference until position 15,000 with approximate ``GQ``
of 40 across the few hundred base pair block.

A sparse matrix table has an entry field ``END`` that corresponds to the GVCF ``INFO`` field,
``END``. It has the same meaning, but only for the single column where the END resides. In a sparse
matrix table, there should be no present entries for this sample between ``chr1:14524`` and
``chr1:15000``, inclusive.

Local Alleles
^^^^^^^^^^^^^

Local alleles are used to reduce the size of arrays such as ``AD``, or most notably ``PL`` at
multi-alleleic sites with many alternate alleles. To illustrate why such a thing is necessary,
consider a site with around 4,000 alleles. A ``PL`` for this site would be ``(4,000 * 4,001)/2
= 8,002,000`` elements long. In a joint cohort of 100,000 samples, if every sample had a ``PL`` at
this position, there would be ``8,002,000 * 100,000 = 800,200,000,000`` elements, each of which is
generally represented as a 4 byte integer, giving a full size of the ``PL`` arrays for this row of
over 3 terabytes. Even if we only had the minimum required ``PL`` arrays materialized, we would
still be looking at gigabytes for a single row.

A sparse matrix table solves this issue by creating new fields that are 'local'. It only stores
information that was present in the imported GVCFs. The :func:`transform_gvcf` does this initial
conversion. The fields ``GT``, ``AD``, ``PGT``, ``PL``, are converted to their local versions,
``LGT``, ``LAD``, ``LPGT``, ``LPL``, and a ``LA`` (local alleles) array is added.  The ``LA`` field
serves as the map between the ``alleles`` field and the local fields. For example (using VCF-like
notation): ::

    LGT:LA:LAD    1/2:0,7,5:0,19,21

For this genotype, the true ``GT`` is ``5/7``, and the depth of the ``5`` Allele is ``21`` and the
depth of the ``7`` allele is ``19``. To get the appropriate ``LPL`` index one can still use
:func:`hail.expr.CallExpression.unphased_diploid_gt_index` of ``LGT``.

Functions
---------

.. autofunction:: combine_gvcfs
.. autofunction:: transform_gvcf
