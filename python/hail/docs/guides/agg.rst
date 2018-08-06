Aggregation
===========

For a full list of aggregators, see the :ref:`aggregators <sec-aggregators>`
section of the API reference.

Table Aggregations
------------------

Aggregate Over Rows Into A Local Value
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Aggregate over the rows of a table using :meth:`.Table.aggregate` and the
:func:`.aggregators.fraction` aggregator. The result is a single value with the
fraction of rows where ``ht.SEX == 'M'``.

>>> ht.aggregate(hl.agg.fraction(ht.SEX == 'M'))
0.5

Perform multiple aggregations using a :class:`.StructExpression` and the
:func:`.aggregators.fraction` and :func:`.aggregators.mean` aggregators. The
result is a struct with two nested fields, ``fraction_male`` and
``mean_x``.

>>> ht.aggregate(hl.struct(fraction_male = hl.agg.fraction(ht.SEX == 'M'),
...                        mean_x = hl.agg.mean(ht.X)))
Struct(fraction_male=0.5, mean_x=6.5)

Aggregate Per Group
~~~~~~~~~~~~~~~~~~~

Group the table ``ht`` by ``ID`` and compute the mean value of ``X`` per group
using the :func:`.aggregators.mean` aggregator.

The method :meth:`.Table.group_by` returns a :class:`.GroupedTable`, on which
we can call the :meth:`.GroupedTable.aggregate` method to aggregate per group.
The final result is a :class:`.Table` with row fields ``ID`` and ``mean_x``.
Other row fields are dropped.

>>> result_ht = ht.group_by(ht.ID).aggregate(mean_x=hl.agg.mean(ht.X))

Matrix Table Aggregations
-------------------------

Aggregate Entries Per Row (Over Columns)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Count the number of occurrences of each unique ``GT`` field per row, i.e.
aggregate over the columns of the matrix table. The count is added as a
row field to the matrix table using :meth:`.MatrixTable.annotate_rows` and
the :func:`.aggregators.counter` aggregator.

Methods :meth:`.MatrixTable.filter_rows`, :meth:`.MatrixTable.select_rows`,
and :meth:`.MatrixTable.transmute_rows` also support aggregation over columns.

>>> result_mt = mt.annotate_rows(gt_counter=hl.agg.counter(mt.GT))

Aggregate Entries Per Column (Over Rows)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Compute the mean of the ``GQ`` field per column, i.e. aggregate over the rows
of the MatrixTable. The result is added as a column field to the matrix table
using :meth:`.MatrixTable.annotate_cols` and the :func:`.aggregators.mean`
aggregator.

Methods :meth:`.MatrixTable.filter_cols`, :meth:`.MatrixTable.select_cols`,
and :meth:`.MatrixTable.transmute_cols` also support aggregation over rows.

>>> result_mt = mt.annotate_cols(gq_mean=hl.agg.mean(mt.GQ))

Aggregate Column Values Into a Local Value
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Aggregate over the column-indexed field ``pheno.is_female`` to compute the
fraction of female samples in the matrix table, using
:meth:`.MatrixTable.aggregate_cols` and the :func:`.aggregators.fraction`
aggregator.

>>> mt.aggregate_cols(hl.agg.fraction(mt.pheno.is_female))
0.5

Aggregate over two column-indexed fields using the :func:`.aggregators.fraction`
and :func:`.aggregators.count_where` aggregators. The result is a single struct
containing two nested fields, ``fraction_female`` and ``case_ratio``.

>>> mt.aggregate_cols(
...             hl.struct(fraction_female=hl.agg.fraction(mt.pheno.is_female),
...                       case_ratio=hl.agg.count_where(mt.is_case) / hl.agg.count()))
Struct(fraction_female=0.5, case_ratio=1.0)

Aggregate Row Values Into a Local Value
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Compute the mean value of the row-indexed field ``qual`` across rows using
:meth:`.MatrixTable.aggregate_rows`, and the :func:`.aggregators.mean`
aggregator. The result is a single value.

>>> mt.aggregate_rows(hl.agg.mean(mt.qual))
404051.99

For the row-indexed field ``qual``, count the number of row values of ``qual``
that are greater than 40, and compute the mean value of ``qual`` across rows,
using :meth:`.MatrixTable.aggregate_rows`. The result is a single struct
containing two nested fields, ``n_high_quality`` and ``mean_qual``.

>>> mt.aggregate_rows(
...             hl.struct(n_high_quality=hl.agg.count_where(mt.qual > 40),
...                       mean_qual=hl.agg.mean(mt.qual)))
Struct(n_high_quality=10, mean_qual=404051.99)

Aggregate Entry Values Into A Local Value
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Compute the mean of the entry-indexed field ``GQ`` and the call rate of
the entry-indexed field ``GT``. The result is returned as a single struct with
two nested fields.

>>> mt.aggregate_entries(
...     hl.struct(global_gq_mean=hl.agg.mean(mt.GQ),
...               call_rate=hl.agg.fraction(hl.is_defined(mt.GT))))
Struct(global_gq_mean=56.73, call_rate=0.976)

Aggregate Per Column Group
~~~~~~~~~~~~~~~~~~~~~~~~~~

Compute the call rate per cohort. Group the columns of the matrix table by
the column-indexed field ``cohort`` using :meth:`.MatrixTable.group_cols_by`,
which returns a :class:`.GroupedMatrixTable`. Then use
:meth:`.GroupedMatrixTable.aggregate` to compute an aggregation per column
group.

The result is a matrix table with an entry field ``call_rate`` that contains
the result of the aggregation. The new matrix table has a row schema equal
to the original row schema, a column schema equal to the fields passed to
``group_cols_by``, and an entry schema determined by the expression passed to
``aggregate``. Other column fields and entry fields are dropped.

>>> result_mt = (mt.group_cols_by(mt.cohort)
...                      .aggregate(call_rate=hl.agg.fraction(hl.is_defined(mt.GT))))

Aggregate Per Row Group
~~~~~~~~~~~~~~~~~~~~~~~

Compute the number of calls with one or more non-reference alleles per
gene group. Group the rows of the matrix table by the row-indexed field ``gene``
using :meth:`.MatrixTable.group_rows_by`, which returns a
:class:`.GroupedMatrixTable`. Then use :meth:`.GroupedMatrixTable.aggregate`
to compute an aggregation per grouped row.

The result is a matrix table with an entry field ``n_non_ref`` that contains
the result of the aggregation. This new matrix table has a row schema
equal to the fields passed to ``group_rows_by``, a column schema equal to the
column schema of the original matrix table, and an entry schema determined
by the expression passed to ``aggregate``. Other row fields and entry fields
are dropped.

>>> result_mt = (mt.group_rows_by(mt.gene)
...                      .aggregate(n_non_ref=hl.agg.count_where(mt.GT.is_non_ref())))



