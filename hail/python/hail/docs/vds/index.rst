.. _sec-vds:

Variant Dataset
===============

The :class:`.VariantDataset` is an extra layer of abstraction of the Hail Matrix Table for working
with large sequencing datasets. It was initially developed in response to the gnomAD project's need
to combine, represent, and analyze 150,000 whole genomes. It has since been used on datasets as
large as 955,000 whole exomes. The :class:`.VariantDatasetCombiner` produces a
:class:`.VariantDataset` by combining any number of GVCF and/or :class:`.VariantDataset` files.

.. warning::

    The :class:`.VariantDataset` API is new and subject to change. While this is functionality tested
    and used in production applications, it is still considered experimental.

.. warning::

    Hail 0.1 also had a Variant Dataset class. Although pieces of the interfaces are similar, they should not
    be considered interchangeable and do not represent the same data.


.. currentmodule:: hail.vds

.. rubric:: Variant Dataset

.. autosummary::
    :nosignatures:
    :toctree: ./
    :template: class2.rst

    VariantDataset

.. autosummary::
    :toctree: ./

    read_vds
    filter_samples
    filter_variants
    filter_intervals
    filter_chromosomes
    sample_qc
    split_multi
    interval_coverage
    impute_sex_chromosome_ploidy
    to_dense_mt
    to_merged_sparse_mt
    truncate_reference_blocks
    merge_reference_blocks
    lgt_to_gt
    local_to_global
    store_ref_block_max_length

.. currentmodule:: hail.vds.combiner

.. rubric:: Variant Dataset Combiner

.. autosummary::
    :nosignatures:
    :toctree: ./
    :template: class2.rst

    VDSMetadata
    VariantDatasetCombiner


.. autosummary::
    :toctree: ./

    new_combiner
    load_combiner

The data model of :class:`.VariantDataset`
------------------------------------------

A VariantDataset is the Hail implementation of a data structure called the
"scalable variant call representation", or SVCR.

The Scalable Variant Call Representation (SVCR)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Like the project VCF (multi-sample VCF) representation, the scalable variant
call representation is a variant-by-sample matrix of records. There are two
fundamental differences, however:

1.  The scalable variant call representation is **sparse**. It is not a dense
    matrix with every entry populated. Reference calls are defined as intervals
    (reference blocks) exactly as they appear in the original GVCFs. Compared to
    a VCF representation, this stores **less data but more information**, and
    makes it possible to keep reference information about every site in the
    genome, not just sites at which there is variation in the current cohort. A
    VariantDataset has a component table of reference information,
    ``vds.reference_data``, which contains the sparse matrix of reference blocks.
    This matrix is keyed by locus (not locus and alleles), and contains an
    ``END`` field which denotes the last position included in the current
    reference block.


2.  The scalable variant call representation uses **local alleles**. In a VCF,
    the fields GT, AD, PL, etc contain information that refers to alleles in the
    VCF by index. At highly multiallelic sites, the number of elements in the
    AD/PL lists explodes to huge numbers, **even though the information content
    does not change**. To avoid this superlinear scaling, the SVCR renames these
    fields to their "local" versions: LGT, LAD, LPL, etc, and adds a new field,
    LA (local alleles). The information in the local fields refers to the alleles
    defined per row of the matrix indirectly through the LA list.

    For instance, if a sample has the following information in its GVCF:

    .. code::

         Ref=G Alt=T GT=0/1 AD=5,6 PL=102,0,150

    If the alternate alleles A,C,T are discovered in the cohort, this sample's
    entry would look like:

    .. code::

         LA=0,2 LGT=0/1 LAD=5,6 LPL=102,0,150

    The "1" allele referred to in LGT, and the allele to which the reads in the
    second position of LAD belong to, is not the allele with absolute index 1
    (**C**), but rather the allele whose index is in position 1 of the LA list.
    The *index* at position 2 of the LA list is 2, and the allele with absolute
    index 2 is **T**. Local alleles make it possible to keep the data small to
    match its inherent information content.

Component tables
^^^^^^^^^^^^^^^^

The :class:`.VariantDataset` is made up of two component matrix tables -- the
``reference_data`` and the ``variant_data``.

The ``reference_data`` matrix table is a sparse matrix of reference blocks. The
``reference_data`` matrix table has row key ``locus``, but
does not have an ``alleles`` key or field. The column key is the sample ID. The
entries indicate regions of reference calls with similar sequencing metadata
(depth, quality, etc), starting from ``vds.reference_data.locus.position`` and
ending at ``vds.reference_data.END`` (inclusive!). There is no ``GT`` call field
because all calls in the reference data are implicitly homozygous reference (in
the future, a table of ploidy by interval may be included to allow for proper
representation of structural variation, but there is no standard representation
for this at current). A record from a component GVCF is included in the
``reference_data`` if it defines the END INFO field (if the GT is not reference,
an error will be thrown by the Hail VDS combiner).


The ``variant_data`` matrix table is a sparse matrix of non-reference calls.
This table contains the complete schema from the component GVCFs, aside from
fields which are known to be defined only for reference blocks (e.g. END or
MIN_DP). A record from a component GVCF is included in the ``variant_data`` if
it does not define the END INFO field. This means that some records of the
``variant_data`` can be no-call (``./.``) or reference, depending on the
semantics of the variant caller that produced the GVCFs.

Building analyses on the :class:`.VariantDataset`
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Analyses operating on sequencing data can be largely grouped into three categories
by functionality used.

1.  **Analyses that use prebuilt methods**. Some analyses can be supported by using
    only the utility functions defined in the ``hl.vds`` module, like
    :func:`.vds.sample_qc`.

2.  **Analyses that use variant data and/or reference data separately.** Some
    pipelines need to interrogate properties of the component tables
    individually. Examples might include singleton analysis or burden tests
    (which needs only to look at the variant data) or coverage analysis (which
    looks only at reference data). These pipelines should explicitly extract and
    manipulate the component tables with ``vds.variant_data`` and
    ``vds.reference_data``.

3.  **Analyses that use the full variant-by-sample matrix with variant and reference data**.
    Many pipelines require variant and reference data together. There are helper
    functions provided for producing either the sparse (containing reference
    blocks) or dense (reference information is filled in at each variant site)
    representations. For more information, see the documentation for
    :func:`.vds.to_dense_mt` and :func:`.vds.to_merged_sparse_mt`.
