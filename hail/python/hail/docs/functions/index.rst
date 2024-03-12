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
    if_else
    switch
    case
    bind
    rbind
    null
    is_missing
    is_defined
    coalesce
    or_else
    or_missing
    range
    query_table

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
    empty_dict

.. rubric:: Collection functions

.. autosummary::

    len
    map
    flatmap
    zip
    enumerate
    zip_with_index
    flatten
    any
    all
    filter
    sorted
    find
    group_by
    fold
    array_scan
    reversed
    keyed_intersection
    keyed_union

.. rubric:: Numeric functions

.. autosummary::

    abs
    approx_equal
    bit_and
    bit_or
    bit_xor
    bit_lshift
    bit_rshift
    bit_not
    bit_count
    exp
    expit
    is_nan
    is_finite
    is_infinite
    log
    log10
    logit
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
    uniroot

.. rubric:: Numeric collection functions

.. autosummary::

    min
    nanmin
    max
    nanmax
    mean
    median
    product
    sum
    cumulative_sum
    argmin
    argmax
    corr
    binary_search

.. rubric:: String functions

.. autosummary::

    format
    json
    parse_json
    hamming
    delimit
    entropy
    parse_int
    parse_int32
    parse_int64
    parse_float
    parse_float32
    parse_float64

.. rubric:: Statistical functions

.. autosummary::

    chi_squared_test
    fisher_exact_test
    contingency_table_test
    cochran_mantel_haenszel_test
    dbeta
    dpois
    hardy_weinberg_test
    pchisqtail
    pnorm
    ppois
    qchisqtail
    qnorm
    qpois

.. rubric:: Randomness

.. autosummary::

    rand_bool
    rand_beta
    rand_cat
    rand_dirichlet
    rand_gamma
    rand_norm
    rand_pois
    rand_unif
    rand_int32
    rand_int64
    shuffle

.. rubric:: Genetics functions

.. autosummary::

    locus
    locus_from_global_position
    locus_interval
    parse_locus
    parse_variant
    parse_locus_interval
    variant_str
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
    contig_length
    allele_type
    pl_dosage
    gp_dosage
    get_sequence
    mendel_error_code
    liftover
    min_rep
    reverse_complement
