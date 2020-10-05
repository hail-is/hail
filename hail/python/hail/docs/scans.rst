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
    +-------+-------+---------+
    |    ID |     Z | cum_sum |
    +-------+-------+---------+
    | int32 | int32 |   int64 |
    +-------+-------+---------+
    |     1 |     4 |       0 |
    |     2 |     3 |       4 |
    |     3 |     3 |       7 |
    |     4 |     2 |      10 |
    +-------+-------+---------+

Note that the cumulative sum is exclusive of the current row's value. On a
matrix table, to compute the cumulative number of non-reference genotype calls
along the genome:

    >>> ds_scan = ds.select_rows(ds.variant_qc.n_non_ref,
    ...                          cum_n_non_ref=hl.scan.sum(ds.variant_qc.n_non_ref))
    >>> ds_scan.rows().show()
    +---------------+------------+-----------+---------------+
    | locus         | alleles    | n_non_ref | cum_n_non_ref |
    +---------------+------------+-----------+---------------+
    | locus<GRCh37> | array<str> |     int64 |         int64 |
    +---------------+------------+-----------+---------------+
    | 20:12990057   | ["T","A"]  |        43 |             0 |
    | 20:13029862   | ["C","T"]  |        99 |            43 |
    | 20:13074235   | ["G","A"]  |        99 |           142 |
    | 20:13140720   | ["G","A"]  |         5 |           241 |
    | 20:13695498   | ["G","A"]  |        25 |           246 |
    | 20:13714384   | ["A","C"]  |         1 |           271 |
    | 20:13765944   | ["C","G"]  |         2 |           272 |
    | 20:13765954   | ["C","T"]  |         2 |           274 |
    | 20:13845987   | ["C","T"]  |       100 |           276 |
    | 20:16223957   | ["T","C"]  |        31 |           376 |
    +---------------+------------+-----------+---------------+
    showing top 10 rows
    <BLANKLINE>

Scans over column fields can be done in a similar manner.

.. DANGER::

    Computing the result of certain aggregators, such as
    :func:`~.aggregators.hardy_weinberg_test`, can be very expensive when done
    for every row in a scan."

See the :ref:`sec-aggregators` module for documentation on the behavior
of specific aggregators.
