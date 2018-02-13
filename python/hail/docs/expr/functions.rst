Functions
=========

.. toctree::
    :maxdepth: 2

.. currentmodule:: hail.expr.functions


Core language functions
-----------------------
.. autosummary::

    broadcast
    capture
    cond
    switch
    case
    bind
    null
    str
    is_missing
    is_defined
    or_else
    or_missing
    range

String functions
----------------
.. autosummary::

    json
    hamming
    delimit

Statistical functions
---------------------
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

Collection constructors
-----------------------
.. autosummary::

    dict
    array
    set

Collection functions
--------------------
.. autosummary::

    map
    flatmap
    flatten
    any
    all
    filter
    sorted
    find
    group_by
    len
    index

Numeric functions
-----------------
.. autosummary::

    exp
    is_nan
    log
    log10
    sqrt


Numeric collection functions
----------------------------
.. autosummary::

    unique_max_index
    unique_min_index
    min
    max
    mean
    median
    product
    sum

Randomness
----------
.. autosummary::

    rand_bool
    rand_norm
    rand_pois
    rand_unif

Genetics functions
------------------
.. autosummary::

    locus
    parse_locus
    parse_variant
    interval
    parse_interval
    call
    unphased_diploid_gt_index_call
    unphased_diploid_gt_index_call
    parse_call
    is_snp
    is_mnp
    is_transition
    is_transversion
    is_insertion
    is_deletion
    is_indel
    is_star
    is_complex
    allele_type
    pl_dosage
    gp_dosage

Core language functions
-----------------------

.. autofunction:: broadcast
.. autofunction:: capture
.. autofunction:: cond
.. autofunction:: switch
.. autofunction:: case
.. autofunction:: bind
.. autofunction:: null
.. autofunction:: str
.. autofunction:: is_missing
.. autofunction:: is_defined
.. autofunction:: or_else
.. autofunction:: or_missing
.. autofunction:: range

String functions
----------------

.. autofunction:: json
.. autofunction:: hamming
.. autofunction:: delimit

Statistical functions
---------------------

.. autofunction:: chisq
.. autofunction:: fisher_exact_test
.. autofunction:: ctt
.. autofunction:: dbeta
.. autofunction:: dpois
.. autofunction:: hardy_weinberg_p
.. autofunction:: pchisqtail
.. autofunction:: pnorm
.. autofunction:: ppois
.. autofunction:: qchisqtail
.. autofunction:: qnorm
.. autofunction:: qpois

Collection constructors
-----------------------

.. autofunction:: dict
.. autofunction:: array
.. autofunction:: set

Collection functions
--------------------

.. autofunction:: map
.. autofunction:: flatmap
.. autofunction:: flatten
.. autofunction:: any
.. autofunction:: all
.. autofunction:: filter
.. autofunction:: sorted
.. autofunction:: find
.. autofunction:: group_by
.. autofunction:: len
.. autofunction:: index

Numeric functions
-----------------

.. autofunction:: exp
.. autofunction:: is_nan
.. autofunction:: log
.. autofunction:: log10
.. autofunction:: sqrt


Numeric collection functions
----------------------------

.. autofunction:: unique_max_index
.. autofunction:: unique_min_index
.. autofunction:: min
.. autofunction:: max
.. autofunction:: mean
.. autofunction:: median
.. autofunction:: product
.. autofunction:: sum

Randomness
----------

.. autofunction:: rand_bool
.. autofunction:: rand_norm
.. autofunction:: rand_pois
.. autofunction:: rand_unif

Genetics functions
------------------

.. autofunction:: locus
.. autofunction:: parse_locus
.. autofunction:: parse_variant
.. autofunction:: interval
.. autofunction:: parse_interval
.. autofunction:: call
.. autofunction:: unphased_diploid_gt_index_call
.. autofunction:: unphased_diploid_gt_index_call
.. autofunction:: parse_call
.. autofunction:: is_snp
.. autofunction:: is_mnp
.. autofunction:: is_transition
.. autofunction:: is_transversion
.. autofunction:: is_insertion
.. autofunction:: is_deletion
.. autofunction:: is_indel
.. autofunction:: is_star
.. autofunction:: is_complex
.. autofunction:: allele_type
.. autofunction:: pl_dosage
.. autofunction:: gp_dosage
