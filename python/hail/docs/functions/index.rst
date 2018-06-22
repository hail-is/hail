.. _sec-functions:

Functions
=========

These functions are exposed at the top level of the module, e.g. ``hl.case``.

.. currentmodule:: hail.expr.functions

.. toctree::
    :maxdepth: 2

    core
    constructors
    collections
    numeric
    string
    stats
    random
    genetics

.. rubric:: Core language functions

.. autosummary::

    literal
    cond
    switch
    case
    bind
    null
    is_missing
    is_defined
    coalesce
    or_else
    or_missing
    range

.. rubric:: Constructors

.. autosummary::

    bool
    float
    float32
    float64
    int
    int32
    int64
    interval
    str
    struct
    tuple

.. rubric:: Collection constructors

.. autosummary::

    array
    empty_array
    set
    empty_set
    dict

.. rubric:: Collection functions

.. autosummary::

    map
    flatmap
    zip
    zip_with_index
    flatten
    any
    all
    filter
    sorted
    find
    group_by
    len

.. rubric:: Numeric functions

.. autosummary::

    abs
    exp
    is_nan
    log
    log10
    sign
    sqrt
    int
    int32
    int64
    float
    float32
    float64
    floor
    ceil

.. rubric:: Numeric collection functions

.. autosummary::

    min
    max
    mean
    median
    product
    sum
    argmin
    argmax

.. rubric:: String functions

.. autosummary::

    json
    hamming
    delimit
    entropy

.. rubric:: Statistical functions

.. autosummary::

    chisq
    fisher_exact_test
    ctt
    dbeta
    dpois
    hardy_weinberg_p
    pchisqtail
    pnorm
    ppois
    qchisqtail
    qnorm
    qpois

.. rubric:: Randomness

.. autosummary::

    rand_bool
    rand_norm
    rand_pois
    rand_unif

.. rubric:: Genetics functions

.. autosummary::

    locus
    locus_from_global_position
    locus_interval
    parse_locus
    parse_variant
    parse_locus_interval
    call
    unphased_diploid_gt_index_call
    parse_call
    downcode
    triangle
    is_snp
    is_mnp
    is_transition
    is_transversion
    is_insertion
    is_deletion
    is_indel
    is_star
    is_complex
    is_valid_contig
    is_valid_locus
    allele_type
    pl_dosage
    gp_dosage
    get_sequence
    mendel_error_code
    liftover
    min_rep
