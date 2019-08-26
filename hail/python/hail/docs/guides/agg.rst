Aggregation
===========

For a full list of aggregators, see the :ref:`aggregators <sec-aggregators>`
section of the API reference.

Table Aggregations
------------------

Aggregate Over Rows Into A Local Value
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

One aggregation
...............

:**description**: Compute the fraction of rows where ``SEX == 'M'`` in a table.

:**code**:

    >>> ht.aggregate(hl.agg.fraction(ht.SEX == 'M'))
    0.5

:**dependencies**: :meth:`.Table.aggregate`, :func:`.aggregators.fraction`

Multiple aggregations
.....................

:**description**: Compute two aggregation statistics, the fraction of rows where
                  ``SEX == 'M'`` and the mean value of ``X``, from the rows of a table.

:**code**:

    >>> ht.aggregate(hl.struct(fraction_male = hl.agg.fraction(ht.SEX == 'M'),
    ...                        mean_x = hl.agg.mean(ht.X)))
    Struct(fraction_male=0.5, mean_x=6.5)

:**dependencies**: :meth:`.Table.aggregate`, :func:`.aggregators.fraction`, :func:`.aggregators.mean`, :class:`.StructExpression`

Aggregate Per Group
~~~~~~~~~~~~~~~~~~~

:**description**: Group the table ``ht`` by ``ID`` and compute the mean value of ``X`` per group.

:**code**:

    >>> result_ht = ht.group_by(ht.ID).aggregate(mean_x=hl.agg.mean(ht.X))

:**dependencies**: :meth:`.Table.group_by`, :meth:`.GroupedTable.aggregate`, :func:`.aggregators.mean`

Matrix Table Aggregations
-------------------------

Aggregate Entries Per Row (Over Columns)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

:**description**:

    Count the number of occurrences of each unique ``GT`` field per row, i.e.
    aggregate over the columns of the matrix table.

    Methods :meth:`.MatrixTable.filter_rows`, :meth:`.MatrixTable.select_rows`,
    and :meth:`.MatrixTable.transmute_rows` also support aggregation over columns.

:**code**:

    >>> result_mt = mt.annotate_rows(gt_counter=hl.agg.counter(mt.GT))

:**dependencies**: :meth:`.MatrixTable.annotate_rows`, :func:`.aggregators.counter`

Aggregate Entries Per Column (Over Rows)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

:**description**:

    Compute the mean of the ``GQ`` field per column, i.e. aggregate over the rows
    of the MatrixTable.

    Methods :meth:`.MatrixTable.filter_cols`, :meth:`.MatrixTable.select_cols`,
    and :meth:`.MatrixTable.transmute_cols` also support aggregation over rows.

:**code**:

    >>> result_mt = mt.annotate_cols(gq_mean=hl.agg.mean(mt.GQ))

:**dependencies**: :meth:`.MatrixTable.annotate_cols`, :func:`.aggregators.mean`

Aggregate Column Values Into a Local Value
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

One aggregation
...............

:**description**:

    Aggregate over the column-indexed field ``pheno.is_female`` to compute the
    fraction of female samples in the matrix table.

:**code**:

    >>> mt.aggregate_cols(hl.agg.fraction(mt.pheno.is_female))
    0.48

:**dependencies**: :meth:`.MatrixTable.aggregate_cols`, :func:`.aggregators.fraction`

Multiple aggregations
.....................

:**description**: Perform multiple aggregations over column-indexed fields by using
                  a struct expression. The result is a single struct containing
                  two nested fields, ``fraction_female`` and ``case_ratio``.

:**code**:

    >>> mt.aggregate_cols(hl.struct(
    ...         fraction_female=hl.agg.fraction(mt.pheno.is_female),
    ...         case_ratio=hl.agg.count_where(mt.is_case) / hl.agg.count()))
    Struct(fraction_female=0.48, case_ratio=1.0)

:**dependencies**: :meth:`.MatrixTable.aggregate_cols`, :func:`.aggregators.fraction`, :func:`.aggregators.count_where`, :class:`.StructExpression`

Aggregate Row Values Into a Local Value
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

One aggregation
...............

:**description**: Compute the mean value of the row-indexed field ``qual``.

:**code**:

    >>> mt.aggregate_rows(hl.agg.mean(mt.qual))
    544323.8915384616

:**dependencies**: :meth:`.MatrixTable.aggregate_rows`, :func:`.aggregators.mean`


Multiple aggregations
.....................

:**description**:

    Perform two row aggregations: count the number of row values of ``qual``
    that are greater than 40, and compute the mean value of ``qual``.
    The result is a single struct containing two nested fields, ``n_high_quality`` and ``mean_qual``.

:**code**:

    >>> mt.aggregate_rows(
    ...             hl.struct(n_high_quality=hl.agg.count_where(mt.qual > 40),
    ...                       mean_qual=hl.agg.mean(mt.qual)))
    Struct(n_high_quality=13, mean_qual=544323.8915384616)

:**dependencies**: :meth:`.MatrixTable.aggregate_rows`, :func:`.aggregators.count_where`, :func:`.aggregators.mean`, :class:`.StructExpression`


Aggregate Entry Values Into A Local Value
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

:**description**:

    Compute the mean of the entry-indexed field ``GQ`` and the call rate of
    the entry-indexed field ``GT``. The result is returned as a single struct with
    two nested fields.

:**code**:

    >>> mt.aggregate_entries(
    ...     hl.struct(global_gq_mean=hl.agg.mean(mt.GQ),
    ...               call_rate=hl.agg.fraction(hl.is_defined(mt.GT))))
    Struct(global_gq_mean=64.01841473178543, call_rate=0.9607692307692308)

:**dependencies**: :meth:`.MatrixTable.aggregate_entries`, :func:`.aggregators.mean`, :func:`.aggregators.fraction`, :class:`.StructExpression`


Aggregate Per Column Group
~~~~~~~~~~~~~~~~~~~~~~~~~~

:**description**: Group the columns of the matrix table by the column-indexed
                  field ``cohort`` and compute the call rate per cohort.

:**code**:

    >>> result_mt = (mt.group_cols_by(mt.cohort)
    ...              .aggregate(call_rate=hl.agg.fraction(hl.is_defined(mt.GT))))

:**dependencies**: :meth:`.MatrixTable.group_cols_by`, :class:`.GroupedMatrixTable`, :meth:`.GroupedMatrixTable.aggregate`

:**understanding**:

    .. container:: toggle

        .. container:: toggle-content

            Group the columns of the matrix table by
            the column-indexed field ``cohort`` using :meth:`.MatrixTable.group_cols_by`,
            which returns a :class:`.GroupedMatrixTable`. Then use
            :meth:`.GroupedMatrixTable.aggregate` to compute an aggregation per column
            group.

            The result is a matrix table with an entry field ``call_rate`` that contains
            the result of the aggregation. The new matrix table has a row schema equal
            to the original row schema, a column schema equal to the fields passed to
            ``group_cols_by``, and an entry schema determined by the expression passed to
            ``aggregate``. Other column fields and entry fields are dropped.

Aggregate Per Row Group
~~~~~~~~~~~~~~~~~~~~~~~

:**description**: Compute the number of calls with one or more non-reference
                  alleles per gene group.

:**code**:

    >>> result_mt = (mt.group_rows_by(mt.gene)
    ...              .aggregate(n_non_ref=hl.agg.count_where(mt.GT.is_non_ref())))

:**dependencies**: :meth:`.MatrixTable.group_rows_by`, :class:`.GroupedMatrixTable`, :meth:`.GroupedMatrixTable.aggregate`

:**understanding**:

    .. container:: toggle

        .. container:: toggle-content

            Group the rows of the matrix table by the row-indexed field ``gene``
            using :meth:`.MatrixTable.group_rows_by`, which returns a
            :class:`.GroupedMatrixTable`. Then use :meth:`.GroupedMatrixTable.aggregate`
            to compute an aggregation per grouped row.

            The result is a matrix table with an entry field ``n_non_ref`` that contains
            the result of the aggregation. This new matrix table has a row schema
            equal to the fields passed to ``group_rows_by``, a column schema equal to the
            column schema of the original matrix table, and an entry schema determined
            by the expression passed to ``aggregate``. Other row fields and entry fields
            are dropped.
