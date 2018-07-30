.. _sec-scan:

Scans
===========

The ``scan`` module is exposed as ``hl.scan``, e.g. ``hl.scan.sum``.

The functions in this module perform rolling aggregations along the rows of a
table, or along the rows or columns of a matrix table. The value of the scan at
a given row (or column) is the result of applying the corresponding aggregator
to all previous rows (or columns). Scans directly over entries are not currently
supported.

For example, the ``count`` aggregator can be used as ``hl.scan.count`` to add an
index along the rows of a table or the rows or columns of a matrix table; the
two statements below produce identical tables:

    >>> ht_with_idx = ht.add_index()
    >>> ht_with_idx = ht.annotate(idx=hl.scan.count())

For example, to compute a cumulative sum for a row field in a table:

    >>> ht_scan = ht.select(ht.ID, ht.Z, cum_sum=hl.scan.sum(ht.Z))
    >>> ht_scan.show()
    +-------+-------+-------------+
    |    ID |     Z | rolling_sum |
    +-------+-------+-------------+
    | int32 | int32 |       int64 |
    +-------+-------+-------------+
    |     1 |     4 |           0 |
    |     2 |     3 |           4 |
    |     3 |     3 |           7 |
    |     4 |     2 |          10 |
    +-------+-------+-------------+

Note that the cumulative sum is exclusive of the current row's value. On a
matrix table, to compute the cumulative number of non-reference genotype calls
along the genome:

    >>> ds_scan = ds.select_rows(ds.variant_qc.n_non_ref,
    ...                          cum_n_non_ref=hl.scan.sum(ds.variant_qc.n_non_ref))
    >>> ds_scan.rows().show()
    +---------------+--------------+-----------+-------------+
    | locus         | alleles      | n_non_ref | rolling_sum |
    +---------------+--------------+-----------+-------------+
    | locus<GRCh37> | array<str>   |     int32 |       int64 |
    +---------------+--------------+-----------+-------------+
    | 20:10579373   | ["C","T"]    |         1 |           0 |
    | 20:13695607   | ["T","G"]    |        23 |           1 |
    | 20:13698129   | ["G","A"]    |         2 |          24 |
    | 20:14306896   | ["G","A"]    |        49 |          26 |
    | 20:14306953   | ["G","T"]    |        62 |          75 |
    | 20:15948325   | ["AG","A"]   |         2 |         137 |
    | 20:15948326   | ["GAAA","G"] |         8 |         139 |
    | 20:17479423   | ["T","C"]    |         1 |         147 |
    | 20:17600357   | ["G","A"]    |        76 |         148 |
    | 20:17640833   | ["A","C"]    |         3 |         224 |
    +---------------+--------------+-----------+-------------+

Scans over column fields can be done in a similar manner.

.. DANGER::

    Computing the result of certain aggregators, such as
    :func:`.agg.hardy_weinberg`, can be very expensive when done for every row
    in a scan."

See the aggregators <aggregators> module for documentation on the behavior
of specific aggregators.