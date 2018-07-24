.. _sec-scan:

Scans
===========

The ``scan`` module is exposed as ``hl.scan``, e.g. ``hl.scan.sum``.

The functions in this module perform rolling aggregations along the desired axis;
For example, the ``count`` aggregator can be used as ``hl.scan.count`` to add an
index along the rows of a `Table` or the rows or columns of a `MatrixTable`.

For example, to add a rolling sum of a Table field, we could do:

    >>> ht_scan = ht.select(ht.ID, ht.Z, rolling_sum=hl.scan.sum(ht.Z))
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

The corresponding calls on MatrixTable fields would be:

    >>> ds_scan = ds.select_rows(ds.variant_qc.n_non_ref,
    ...                          rolling_sum=hl.scan.sum(ds.variant_qc.n_non_ref))
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

    >>> ds_scan = ds.select_cols(ds.sample_qc.n_non_ref,
    ...                          rolling_sum=hl.scan.sum(ds.sample_qc.n_non_ref))
    >>> ds_scan.cols().show()
    +----------------+-----------+-------------+
    | s              | n_non_ref | rolling_sum |
    +----------------+-----------+-------------+
    | str            |     int64 |       int64 |
    +----------------+-----------+-------------+
    | C1046::HG02024 |         3 |           0 |
    | C1046::HG02025 |         3 |           3 |
    | C1046::HG02026 |         2 |           6 |
    | C1047::HG00731 |         3 |           8 |
    | C1047::HG00732 |         1 |          11 |
    | C1047::HG00733 |         3 |          12 |
    | C1048::HG02024 |         3 |          15 |
    | C1048::HG02025 |         3 |          18 |
    | C1048::HG02026 |         2 |          21 |
    | C1049::HG00731 |         2 |          23 |
    +----------------+-----------+-------------+


Scans over entry fields are currently not supported.

.. DANGER::

    Computing the result of certain aggregators, such as ``hardy_weinberg``, can
    be expensive.

See the ``aggregators` `<aggregators> module for documentation on the behavior
of specific aggregators.