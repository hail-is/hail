import itertools
from typing import *
from collections import OrderedDict
import warnings

import hail
import hail as hl
from hail.expr.expressions import *
from hail.expr.types import *
from hail.ir import *
from hail.table import Table, ExprContainer
from hail.typecheck import *
from hail.utils import storage_level, LinkedList
from hail.utils.java import escape_id, warn, jiterable_to_list, Env, scala_object, joption, jnone
from hail.utils.misc import *


class GroupedMatrixTable(ExprContainer):
    """Matrix table grouped by row or column that can be aggregated into a new matrix table."""

    def __init__(self, parent: 'MatrixTable', row_keys=None, col_keys=None, entry_fields=None, row_fields=None, col_fields=None):
        super(GroupedMatrixTable, self).__init__()
        self._parent = parent
        self._copy_fields_from(parent)
        self._row_keys = row_keys
        self._col_keys = col_keys
        self._entry_fields = entry_fields
        self._row_fields = row_fields
        self._col_fields = col_fields
        self._partitions = None

    def _fixed_indices(self):
        if self._row_keys is None and self._col_keys is None:
            return self._parent._entry_indices
        elif self._row_keys is not None and self._col_keys is None:
            return self._parent._col_indices
        elif self._row_keys is None and self._col_keys is not None:
            return self._parent._row_indices
        else:
            return self._parent._global_indices

    @typecheck_method(item=str)
    def __getitem__(self, item):
        return self._get_field(item)

    def describe(self, handler=print):
        """Print information about grouped matrix table."""

        if self._row_keys is None:
            rowstr = ""
        else:
            rowstr = "\nRows: \n" + "\n    ".join(["{}: {}".format(k, v._type) for k, v in self._row_keys.items()])

        if self._col_keys is None:
            colstr = ""
        else:
            colstr = "\nColumns: \n" + "\n    ".join(["{}: {}".format(k, v) for k, v in self._col_keys.items()])

        s = '----------------------------------------\n' \
            'GroupedMatrixTable grouped by {}\n' \
            '----------------------------------------\n' \
            'Parent MatrixTable:\n'.format(
            rowstr,
            colstr)

        handler(s)
        self._parent.describe(handler)

    @typecheck_method(exprs=oneof(str, Expression),
                      named_exprs=expr_any)
    def group_rows_by(self, *exprs, **named_exprs) -> 'GroupedMatrixTable':
        """Group rows.

        Examples
        --------
        Aggregate to a matrix with genes as row keys, computing the number of
        non-reference calls as an entry field:

        >>> dataset_result = (dataset.group_rows_by(dataset.gene)
        ...                          .aggregate(n_non_ref = agg.count_where(dataset.GT.is_non_ref())))

        Notes
        -----
        All complex expressions must be passed as named expressions.

        Parameters
        ----------
        exprs : args of :obj:`str` or :class:`.Expression`
            Row fields to group by.
        named_exprs : keyword args of :class:`.Expression`
            Row-indexed expressions to group by.

        Returns
        -------
        :class:`.GroupedMatrixTable`
            Grouped matrix. Can be used to call :meth:`.GroupedMatrixTable.aggregate`.
        """
        if self._row_keys:
            raise NotImplementedError("GroupedMatrixTable is already grouped by rows.")
        if self._col_keys:
            raise NotImplementedError("GroupedMatrixTable is already grouped by cols; cannot also group by rows.")
        new_keys = {}
        kept_fields = list(self._parent.globals.dtype)
        if self._col_keys is None:
            kept_fields.extend(list(self._parent.col.dtype))

        for e in exprs:
            if isinstance(e, str):
                e = self[e]
            else:
                e = to_expr(e)
            analyze('MatrixTable.group_rows_by', e, self._parent._row_indices)
            if not e._ir.is_nested_field:
                raise ExpressionException("method 'group_rows_by' expects keyword arguments for complex expressions")
            key = e._ir.name

            if key in new_keys or key in kept_fields:
                raise ExpressionException("method 'group_rows_by' found duplicate field: {}".format(key))
            new_keys[key] = e

        for key, e in named_exprs.items():
            if key in new_keys or key in kept_fields:
                raise ExpressionException("method 'group_rows_by' found duplicate field: {}".format(key))
            new_keys[key] = e

        return GroupedMatrixTable(self._parent, row_keys=new_keys)

    @typecheck_method(exprs=oneof(str, Expression),
                      named_exprs=expr_any)
    def group_cols_by(self, *exprs, **named_exprs) -> 'GroupedMatrixTable':
        """Group columns.

        Examples
        --------
        Aggregate to a matrix with cohort as column keys, computing the call rate
        as an entry field:

        >>> dataset_result = (dataset.group_cols_by(dataset.cohort)
        ...                          .aggregate(call_rate = agg.fraction(hl.is_defined(dataset.GT))))

        Notes
        -----
        All complex expressions must be passed as named expressions.

        Parameters
        ----------
        exprs : args of :obj:`str` or :class:`.Expression`
            Column fields to group by.
        named_exprs : keyword args of :class:`.Expression`
            Column-indexed expressions to group by.

        Returns
        -------
        :class:`.GroupedMatrixTable`
            Grouped matrix, can be used to call :meth:`.GroupedMatrixTable.aggregate`.
        """
        if self._row_keys:
            raise NotImplementedError("GroupedMatrixTable is already grouped by rows; cannot also group by cols.")
        if self._col_keys:
            raise NotImplementedError("GroupedMatrixTable is already grouped by cols.")
        new_keys = {}
        kept_fields = list(self._parent.globals.dtype)
        if self._row_keys is None:
            kept_fields.extend(list(self._parent.row.dtype))

        for e in exprs:
            if isinstance(e, str):
                e = self[e]
            else:
                e = to_expr(e)
            analyze('MatrixTable.group_cols_by', e, self._parent._col_indices)
            if not e._ir.is_nested_field:
                raise ExpressionException("method 'group_cols_by' expects keyword arguments for complex expressions")
            key = e._ir.name
            if key in new_keys or key in kept_fields:
                raise ExpressionException("method 'group_cols_by' found duplicate field: {}".format(key))
            new_keys[key] = e

        for key, e in named_exprs.items():
            if key in new_keys or key in kept_fields:
                raise ExpressionException("method 'group_cols_by' found duplicate field: {}".format(key))
            new_keys[key] = e

        return GroupedMatrixTable(self._parent, col_keys=new_keys)

    def partition_hint(self, n: int) -> 'GroupedMatrixTable':
        """Set the target number of partitions for aggregation.

        Examples
        --------

        Use `partition_hint` in a :meth:`.MatrixTable.group_rows_by` /
        :meth:`.GroupedMatrixTable.aggregate` pipeline:

        >>> dataset_result = (dataset.group_rows_by(dataset.gene)
        ...                          .partition_hint(5)
        ...                          .aggregate(n_non_ref = agg.count_where(dataset.GT.is_non_ref())))

        Notes
        -----
        Until Hail's query optimizer is intelligent enough to sample records at all
        stages of a pipeline, it can be necessary in some places to provide some
        explicit hints.

        The default number of partitions for :meth:`.GroupedMatrixTable.aggregate` is
        the number of partitions in the upstream dataset. If the aggregation greatly
        reduces the size of the dataset, providing a hint for the target number of
        partitions can accelerate downstream operations.

        Parameters
        ----------
        n : int
            Number of partitions.

        Returns
        -------
        :class:`.GroupedMatrixTable`
            Same grouped matrix table with a partition hint.
        """

        self._partitions = n
        return self

    @typecheck_method(named_exprs=expr_any)
    def aggregate_cols(self, **named_exprs) -> 'GroupedMatrixTable':
        """Aggregate cols by group.

        Examples
        --------
        Aggregate to a matrix with cohort as column keys, computing the mean height
        per cohort as a new column field:

        >>> dataset_result = (dataset.group_cols_by(dataset.cohort)
        ...                          .aggregate_cols(mean_height = agg.mean(dataset.pheno.height))
        ...                          .result())

        Notes
        -----
        The aggregation scope includes all column fields and global fields.

        See Also
        --------
        :meth:`.result`

        Parameters
        ----------
        named_exprs : varargs of :class:`.Expression`
            Aggregation expressions.

        Returns
        -------
        :class:`.GroupedMatrixTable`
        """

        if self._row_keys:
            raise NotImplementedError("GroupedMatrixTable is already grouped by rows. Cannot aggregate over cols.")

        assert self._col_keys is not None

        existing_fields = list(self._parent.globals.dtype)
        existing_fields.extend(self._col_keys.keys())
        existing_fields.extend(list(self._parent.row.dtype))
        if self._entry_fields is not None:
            existing_fields.extend(list(self._entry_fields.keys()))

        new_fields = self._col_fields if self._col_fields is not None else {}
        for k, e in named_exprs.items():
            if k in existing_fields or k in new_fields:
                raise ExpressionException(f"GroupedMatrixTable.aggregate_cols cannot assign duplicate field {repr(k)}")
            analyze('GroupedMatrixTable.aggregate_cols', e, self._parent._global_indices, {self._parent._col_axis})
            new_fields[k] = e

        return GroupedMatrixTable(self._parent,
                                  row_keys=self._row_keys,
                                  col_keys=self._col_keys,
                                  entry_fields=self._entry_fields,
                                  row_fields=self._row_fields,
                                  col_fields=new_fields)

    @typecheck_method(named_exprs=expr_any)
    def aggregate_rows(self, **named_exprs) -> 'GroupedMatrixTable':
        """Aggregate rows by group.

        Examples
        --------
        Aggregate to a matrix with genes as row keys, collecting the functional
        consequences per gene as a set as a new row field:

        >>> dataset_result = (dataset.group_rows_by(dataset.gene)
        ...                          .aggregate_rows(consequences = agg.collect_as_set(dataset.consequence))
        ...                          .result())

        Notes
        -----
        The aggregation scope includes all row fields and global fields.

        See Also
        --------
        :meth:`.result`

        Parameters
        ----------
        named_exprs : varargs of :class:`.Expression`
            Aggregation expressions.

        Returns
        -------
        :class:`.GroupedMatrixTable`
        """

        if self._col_keys:
            raise NotImplementedError("GroupedMatrixTable is already grouped by cols. Cannot aggregate over rows.")

        assert self._row_keys is not None

        existing_fields = list(self._parent.globals.dtype)
        existing_fields.extend(self._row_keys.keys())
        existing_fields.extend(list(self._parent.col.dtype))
        if self._entry_fields is not None:
            existing_fields.extend(list(self._entry_fields.keys()))

        new_fields = self._row_fields if self._row_fields is not None else {}
        for k, e in named_exprs.items():
            if k in existing_fields or k in new_fields:
                raise ExpressionException(f"GroupedMatrixTable.aggregate_rows cannot assign duplicate field {repr(k)}")
            analyze('GroupedMatrixTable.aggregate_rows', e, self._parent._global_indices, {self._parent._row_axis})
            new_fields[k] = e

        return GroupedMatrixTable(self._parent,
                                  row_keys=self._row_keys,
                                  col_keys=self._col_keys,
                                  entry_fields=self._entry_fields,
                                  row_fields=new_fields,
                                  col_fields=self._col_fields)

    @typecheck_method(named_exprs=expr_any)
    def aggregate_entries(self, **named_exprs) -> 'GroupedMatrixTable':
        """Aggregate entries by group.

        Examples
        --------
        Aggregate to a matrix with genes as row keys, computing the number of
        non-reference calls as an entry field:

        >>> dataset_result = (dataset.group_rows_by(dataset.gene)
        ...                          .aggregate_entries(n_non_ref = agg.count_where(dataset.GT.is_non_ref()))
        ...                          .result())

        See Also
        --------
        :meth:`.aggregate`, :meth:`.result`

        Parameters
        ----------
        named_exprs : varargs of :class:`.Expression`
            Aggregation expressions.

        Returns
        -------
        :class:`.GroupedMatrixTable`
        """
        assert self._row_keys is not None or self._col_keys is not None

        fixed_fields = list(self._parent.globals.dtype)

        if self._row_keys is not None:
            fixed_fields.extend(self._row_keys.keys())
            if self._row_fields is not None:
                fixed_fields.extend(list(self._row_fields.keys()))
        else:
            fixed_fields.extend(list(self._parent.row.dtype))

        if self._col_keys is not None:
            fixed_fields.extend(self._col_keys.keys())
            if self._col_fields is not None:
                fixed_fields.extend(list(self._col_fields.keys()))
        else:
            fixed_fields.extend(list(self._parent.col.dtype))

        new_fields = self._entry_fields if self._entry_fields is not None else {}
        for k, e in named_exprs.items():
            if k in fixed_fields or k in new_fields:
                raise ExpressionException(f"GroupedMatrixTable.aggregate_entries cannot assign duplicate field {repr(k)}")
            analyze('GroupedMatrixTable.aggregate_entries', e, self._fixed_indices(), {self._parent._row_axis, self._parent._col_axis})
            new_fields[k] = e

        return GroupedMatrixTable(self._parent,
                                  row_keys=self._row_keys,
                                  col_keys=self._col_keys,
                                  entry_fields=new_fields,
                                  row_fields=self._row_fields,
                                  col_fields=self._col_fields)

    def result(self) -> 'MatrixTable':
        """Return the result of aggregating by group.

        Examples
        --------
        Aggregate to a matrix with genes as row keys, collecting the functional
        consequences per gene as a row field and computing the number of
        non-reference calls as an entry field:

        >>> dataset_result = (dataset.group_rows_by(dataset.gene)
        ...                          .aggregate_rows(consequences = agg.collect_as_set(dataset.consequence))
        ...                          .aggregate_entries(n_non_ref = agg.count_where(dataset.GT.is_non_ref()))
        ...                          .result())

        Aggregate to a matrix with cohort as column keys, computing the mean height
        per cohort as a column field and computing the number of non-reference calls
        as an entry field:

        >>> dataset_result = (dataset.group_cols_by(dataset.cohort)
        ...                          .aggregate_cols(mean_height = agg.stats(dataset.pheno.height).mean)
        ...                          .aggregate_entries(n_non_ref = agg.count_where(dataset.GT.is_non_ref()))
        ...                          .result())

        See Also
        --------
        :meth:`.aggregate`

        Returns
        -------
        :class:`.MatrixTable`
            Aggregated matrix table.
        """
        if self._col_keys is None and self._row_keys is None:
            raise ValueError("GroupedMatrixTable cannot be aggregated if no groupings are specified.")

        group_exprs = dict(self._col_keys) if self._col_keys is not None else dict(self._row_keys)
        entry_exprs = dict(self._entry_fields) if self._entry_fields is not None else {}
        row_exprs = dict(self._row_fields) if self._row_fields is not None else {}
        col_exprs = dict(self._col_fields) if self._col_fields is not None else {}

        if len(entry_exprs) == 0:
            warn("'GroupedMatrixTable.result': No entry fields were defined.")

        base, cleanup = self._parent._process_joins(*group_exprs.values(),
                                                    *entry_exprs.values(),
                                                    *row_exprs.values(),
                                                    *col_exprs.values())

        if self._col_keys is not None:
            keyed_mt = base._select_cols_processed(hl.struct(**group_exprs))
            mt = MatrixTable(MatrixAggregateColsByKey(keyed_mt._mir,
                                                      hl.struct(**entry_exprs)._ir,
                                                      hl.struct(**col_exprs)._ir))
        else:
            assert self._row_keys is not None
            keyed_mt = base._select_rows_processed(hl.struct(**group_exprs))
            mt = MatrixTable(MatrixAggregateRowsByKey(keyed_mt._mir,
                                                      hl.struct(**entry_exprs)._ir,
                                                      hl.struct(**row_exprs)._ir))

        return cleanup(mt)

    @typecheck_method(named_exprs=expr_any)
    def aggregate(self, **named_exprs) -> 'MatrixTable':
        """Aggregate entries by group, used after :meth:`.MatrixTable.group_rows_by`
        or :meth:`.MatrixTable.group_cols_by`.

        Examples
        --------
        Aggregate to a matrix with genes as row keys, computing the number of
        non-reference calls as an entry field:

        >>> dataset_result = (dataset.group_rows_by(dataset.gene)
        ...                          .aggregate(n_non_ref = agg.count_where(dataset.GT.is_non_ref())))

        Notes
        -----
        Alias for :meth:`aggregate_entries`, :meth:`result`.

        See Also
        --------
        :meth:`aggregate_entries`, :meth:`result`

        Parameters
        ----------
        named_exprs : varargs of :class:`.Expression`
            Aggregation expressions.

        Returns
        -------
        :class:`.MatrixTable`
            Aggregated matrix table.
        """

        return self.aggregate_entries(**named_exprs).result()


matrix_table_type = lazy()


class MatrixTable(ExprContainer):
    """Hail's distributed implementation of a structured matrix.

    Use :func:`.read_matrix_table` to read a matrix table that was written with
    :meth:`.MatrixTable.write`.

    Examples
    --------

    Add annotations:

    >>> dataset = dataset.annotate_globals(pli={'SCN1A': 0.999, 'SONIC': 0.014},
    ...                                    populations = ['AFR', 'EAS', 'EUR', 'SAS', 'AMR', 'HIS'])

    >>> dataset = dataset.annotate_cols(pop = dataset.populations[hl.int(hl.rand_unif(0, 6))],
    ...                                 sample_gq = agg.mean(dataset.GQ),
    ...                                 sample_dp = agg.mean(dataset.DP))

    >>> dataset = dataset.annotate_rows(variant_gq = agg.mean(dataset.GQ),
    ...                                 variant_dp = agg.mean(dataset.GQ),
    ...                                 sas_hets = agg.count_where(dataset.GT.is_het()))

    >>> dataset = dataset.annotate_entries(gq_by_dp = dataset.GQ / dataset.DP)

    Filter:

    >>> dataset = dataset.filter_cols(dataset.pop != 'EUR')

    >>> datasetm = dataset.filter_rows((dataset.variant_gq > 10) & (dataset.variant_dp > 5))

    >>> dataset = dataset.filter_entries(dataset.gq_by_dp > 1)

    Query:

    >>> col_stats = dataset.aggregate_cols(hl.struct(pop_counts=agg.counter(dataset.pop),
    ...                                              high_quality=agg.fraction((dataset.sample_gq > 10) & (dataset.sample_dp > 5))))
    >>> print(col_stats.pop_counts)
    >>> print(col_stats.high_quality)

    >>> het_dist = dataset.aggregate_rows(agg.stats(dataset.sas_hets))
    >>> print(het_dist)

    >>> entry_stats = dataset.aggregate_entries(hl.struct(call_rate=agg.fraction(hl.is_defined(dataset.GT)),
    ...                                                   global_gq_mean=agg.mean(dataset.GQ)))
    >>> print(entry_stats.call_rate)
    >>> print(entry_stats.global_gq_mean)
    """

    @staticmethod
    def _from_java(jmt):
        return MatrixTable(JavaMatrix(jmt.ast()))

    def __init__(self, mir):
        super(MatrixTable, self).__init__()

        self._mir = mir
        self._jmir = mir.to_java_ir()
        self._jmt = Env.hail().variant.MatrixTable(Env.hc()._jhc, self._jmir)

        jmtype = self._jmir.typ()

        self._globals = None
        self._col_values = None

        self._row_axis = 'row'
        self._col_axis = 'column'

        self._global_indices = Indices(self, set())
        self._row_indices = Indices(self, {self._row_axis})
        self._col_indices = Indices(self, {self._col_axis})
        self._entry_indices = Indices(self, {self._row_axis, self._col_axis})

        self._global_type = hl.dtype(jmtype.globalType().toString())
        self._col_type = hl.dtype(jmtype.colType().toString())
        self._row_type = hl.dtype(jmtype.rowType().toString())
        self._entry_type = hl.dtype(jmtype.entryType().toString())

        assert isinstance(self._global_type, tstruct), self._global_type
        assert isinstance(self._col_type, tstruct), self._col_type
        assert isinstance(self._row_type, tstruct), self._row_type
        assert isinstance(self._entry_type, tstruct), self._entry_type

        self._globals = construct_reference('global', self._global_type,
                                            indices=self._global_indices)
        self._rvrow = construct_reference('va',
                                          hl.dtype(jmtype.rvRowType().toString()),
                                          indices=self._row_indices)
        self._row = hail.struct(**{k: self._rvrow[k] for k in self._row_type.keys()})
        self._col = construct_reference('sa', self._col_type,
                                        indices=self._col_indices)
        self._entry = construct_reference('g', self._entry_type,
                                          indices=self._entry_indices)

        self._indices_from_ref = {'global': self._global_indices,
                                  'va': self._row_indices,
                                  'sa': self._col_indices,
                                  'g': self._entry_indices}

        self._row_key = hail.struct(
            **{k: self._row[k] for k in jiterable_to_list(jmtype.rowKey())})
        self._partition_key = self._row_key
        self._col_key = hail.struct(
            **{k: self._col[k] for k in jiterable_to_list(jmtype.colKey())})

        self._num_samples = None

        for k, v in itertools.chain(self._globals.items(),
                                    self._row.items(),
                                    self._col.items(),
                                    self._entry.items()):
            self._set_field(k, v)

    def __getitem__(self, item):
        invalid_usage = TypeError(f"MatrixTable.__getitem__: invalid index argument(s)\n"
                                  f"  Usage 1: field selection ( mt['field'] )\n"
                                  f"  Usage 2: Entry joining ( mt[mt2.row_key, mt2.col_key] )")

        if isinstance(item, str):
            return self._get_field(item)
        elif isinstance(item, tuple) and len(item) == 2:
            # this is the join path
            exprs = item
            row_key = wrap_to_tuple(exprs[0])
            col_key = wrap_to_tuple(exprs[1])

            try:
                return self.index_entries(row_key, col_key)
            except TypeError as e:
                raise invalid_usage from e
        else:
            raise invalid_usage

    @property
    def col_key(self):
        """Column key struct.

        Examples
        --------

        Get the column key field names:

        >>> list(dataset.col_key)
        ['s']

        Returns
        -------
        :class:`.StructExpression`
        """
        return self._col_key

    @property
    def row_key(self):
        """Row key struct.

        Examples
        --------

        Get the row key field names:

        >>> list(dataset.row_key)
        ['locus', 'alleles']

        Returns
        -------
        :class:`.StructExpression`
        """
        return self._row_key

    @property
    def globals(self):
        """Returns a struct expression including all global fields.

        Returns
        -------
        :class:`.StructExpression`
        """
        return self._globals

    @property
    def row(self):
        """Returns a struct expression of all row-indexed fields, including keys.

        Examples
        --------
        Get the first five row field names:

        >>> list(dataset.row)[:5]
        ['locus', 'alleles', 'rsid', 'qual', 'filters']

        Returns
        -------
        :class:`.StructExpression`
            Struct of all row fields.
        """
        return self._row

    @property
    def row_value(self):
        """Returns a struct expression including all non-key row-indexed fields.

        Examples
        --------
        Get the first five non-key row field names:

            >>> list(dataset.row_value)[:5]
            ['rsid', 'qual', 'filters', 'info', 'use_as_marker']

        Returns
        -------
        :class:`.StructExpression`
            Struct of all row fields, minus keys.
        """
        return self._row.drop(*self.row_key)

    @property
    def col(self):
        """Returns a struct expression of all column-indexed fields, including keys.

        Examples
        --------
        Get all column field names:

        >>> list(dataset.col)  # doctest: +NOTEST
        ['s', 'sample_qc', 'is_case', 'pheno', 'cov', 'cov1', 'cov2', 'cohorts', 'pop']

        Returns
        -------
        :class:`.StructExpression`
            Struct of all column fields.
        """
        return self._col

    @property
    def col_value(self):
        """Returns a struct expression including all non-key column-indexed fields.

        Examples
        --------
        Get all non-key column field names:

        >>> list(dataset.col_value)  # doctest: +NOTEST
        ['sample_qc', 'is_case', 'pheno', 'cov', 'cov1', 'cov2', 'cohorts', 'pop']

        Returns
        -------
        :class:`.StructExpression`
            Struct of all column fields, minus keys.
        """
        return self._col.drop(*self.col_key)

    @property
    def entry(self):
        """Returns a struct expression including all row-and-column-indexed fields.

        Examples
        --------
        Get all entry field names:

        >>> list(dataset.entry)
        ['GT', 'AD', 'DP', 'GQ', 'PL']


        Returns
        -------
        :class:`.StructExpression`
            Struct of all entry fields.
        """
        return self._entry

    @typecheck_method(keys=oneof(str, Expression),
                      named_keys=expr_any)
    def key_cols_by(self, *keys, **named_keys) -> 'MatrixTable':
        """Key columns by a new set of fields.

        See :meth:`.Table.key_by` for more information on defining a key.

        Parameters
        ----------
        keys : varargs of :obj:`str` or :class:`.Expression`.
            Column fields to key by.
        named_keys : keyword args of :class:`.Expression`.
            Column fields to key by.
        Returns
        -------
        :class:`.MatrixTable`
        """
        key_fields = get_select_exprs("MatrixTable.key_cols_by",
                                      keys, named_keys, self._col_indices,
                                      protect_keys=False)
        return self._select_cols("MatrixTable.key_cols_by",
                                 self.col.annotate(**key_fields),
                                 new_key=list(key_fields.keys()))

    @typecheck_method(keys=oneof(str, Expression),
                      named_keys=expr_any)
    def key_rows_by(self, *keys, **named_keys) -> 'MatrixTable':
        """Key rows by a new set of fields.

        Examples
        --------

        >>> dataset_result = dataset.key_rows_by('locus')
        >>> dataset_result = dataset.key_rows_by(dataset['locus'])
        >>> dataset_result = dataset.key_rows_by(**dataset.row_key.drop('alleles'))

        All of these expressions key the dataset by the 'locus' field, dropping
        the 'alleles' field from the row key.

        >>> dataset_result = dataset.key_rows_by(contig=dataset['locus'].contig,
        ...                                      position=dataset['locus'].position,
        ...                                      alleles=dataset['alleles'])

        This keys the dataset by the newly defined fields, 'contig' and 'position',
        and the 'alleles' field. The old row key field, 'locus', is preserved as
        a non-key field.

        Notes
        -----
        See :meth:`.Table.key_by` for more information on defining a key.

        Parameters
        ----------
        keys : varargs of :obj:`str` or :class:`.Expression`.
            Row fields to key by.
        named_keys : keyword args of :class:`.Expression`.
            Row fields to key by.
        Returns
        -------
        :class:`.MatrixTable`
        """

        key_fields = get_select_exprs("MatrixTable.key_rows_by",
                                      keys, named_keys, self._row_indices,
                                      protect_keys=False)

        new_row = self._rvrow.annotate(**key_fields)
        base, cleanup = self._process_joins(new_row)

        return cleanup(MatrixTable(
            MatrixKeyRowsBy(
                MatrixMapRows(
                    MatrixKeyRowsBy(base._mir, []),
                    new_row._ir),
                list(key_fields))))

    def annotate_globals(self, **named_exprs) -> 'MatrixTable':
        """Create new global fields by name.

        Examples
        --------
        Add two global fields:

        >>> pops_1kg = {'EUR', 'AFR', 'EAS', 'SAS', 'AMR'}
        >>> dataset_result = dataset.annotate_globals(pops_in_1kg = pops_1kg,
        ...                                           gene_list = ['SHH', 'SCN1A', 'SPTA1', 'DISC1'])

        Add global fields from another table and matrix table:

        >>> dataset_result = dataset.annotate_globals(thing1 = dataset2.index_globals().global_field,
        ...                                           thing2 = v_metadata.index_globals().global_field)

        Note
        ----
        This method does not support aggregation.

        Notes
        -----
        This method creates new global fields, but can also overwrite existing fields. Only
        same-scope fields can be overwritten: for example, it is not possible to annotate a
        row field `foo` and later create an global field `foo`. However, it would be possible
        to create an global field `foo` and later create another global field `foo`, overwriting
        the first.

        The arguments to the method should either be :class:`.Expression`
        objects, or should be implicitly interpretable as expressions.

        Parameters
        ----------
        named_exprs : keyword args of :class:`.Expression`
            Field names and the expressions to compute them.

        Returns
        -------
        :class:`.MatrixTable`
            Matrix table with new global field(s).
        """

        named_exprs = {k: to_expr(v) for k, v in named_exprs.items()}

        for k, v in named_exprs.items():
            check_collisions(self._fields, k, self._global_indices)

        return self._select_globals("MatrixTable.annotate_globals", self.globals.annotate(**named_exprs))

    def annotate_rows(self, **named_exprs) -> 'MatrixTable':
        """Create new row-indexed fields by name.

        Examples
        --------
        Compute call statistics for high quality samples per variant:

        >>> high_quality_calls = agg.filter(dataset.sample_qc.gq_stats.mean > 20,
        ...                                 agg.call_stats(dataset.GT, dataset.alleles))
        >>> dataset_result = dataset.annotate_rows(call_stats = high_quality_calls)

        Add functional annotations from a :class:`.Table` keyed by :class:`.TVariant`:, and another
        :class:`.MatrixTable`.

        >>> dataset_result = dataset.annotate_rows(consequence = v_metadata[dataset.locus, dataset.alleles].consequence,
        ...                                        dataset2_AF = dataset2.index_rows(dataset.row_key).info.AF)

        Note
        ----
        This method supports aggregation over columns. For instance, the usage:

        >>> dataset_result = dataset.annotate_rows(mean_GQ = agg.mean(dataset.GQ))

        will compute the mean per row.

        Notes
        -----
        This method creates new row fields, but can also overwrite existing fields. Only
        non-key, same-scope fields can be overwritten: for example, it is not possible
        to annotate a global field `foo` and later create an row field `foo`. However,
        it would be possible to create an row field `foo` and later create another row
        field `foo`, overwriting the first, as long as `foo` is not a row key.

        The arguments to the method should either be :class:`.Expression`
        objects, or should be implicitly interpretable as expressions.

        Parameters
        ----------
        named_exprs : keyword args of :class:`.Expression`
            Field names and the expressions to compute them.

        Returns
        -------
        :class:`.MatrixTable`
            Matrix table with new row-indexed field(s).
        """

        caller = "MatrixTable.annotate_rows"
        e = get_annotate_exprs(caller, named_exprs, self._row_indices)
        return self._select_rows(caller, self._rvrow.annotate(**e))

    def annotate_cols(self, **named_exprs) -> 'MatrixTable':
        """Create new column-indexed fields by name.

        Examples
        --------
        Compute statistics about the GQ distribution per sample:

        >>> dataset_result = dataset.annotate_cols(sample_gq_stats = agg.stats(dataset.GQ))

        Add sample metadata from a :class:`.hail.Table`.

        >>> dataset_result = dataset.annotate_cols(population = s_metadata[dataset.s].pop)

        Note
        ----
        This method supports aggregation over rows. For instance, the usage:

        >>> dataset_result = dataset.annotate_cols(mean_GQ = agg.mean(dataset.GQ))

        will compute the mean per column.

        Notes
        -----
        This method creates new column fields, but can also overwrite existing fields. Only
        same-scope fields can be overwritten: for example, it is not possible to annotate a
        global field `foo` and later create an column field `foo`. However, it would be possible
        to create an column field `foo` and later create another column field `foo`, overwriting
        the first.

        The arguments to the method should either be :class:`.Expression`
        objects, or should be implicitly interpretable as expressions.

        Parameters
        ----------
        named_exprs : keyword args of :class:`.Expression`
            Field names and the expressions to compute them.

        Returns
        -------
        :class:`.MatrixTable`
            Matrix table with new column-indexed field(s).
        """
        caller = "MatrixTable.annotate_cols"
        e = get_annotate_exprs(caller, named_exprs, self._col_indices)
        return self._select_cols(caller, self.col.annotate(**e))

    def annotate_entries(self, **named_exprs) -> 'MatrixTable':
        """Create new row-and-column-indexed fields by name.

        Examples
        --------
        Compute the allele dosage using the PL field:

        >>> def get_dosage(pl):
        ...    # convert to linear scale
        ...    linear_scaled = pl.map(lambda x: 10 ** - (x / 10))
        ...
        ...    # normalize to sum to 1
        ...    ls_sum = hl.sum(linear_scaled)
        ...    linear_scaled = linear_scaled.map(lambda x: x / ls_sum)
        ...
        ...    # multiply by [0, 1, 2] and sum
        ...    return hl.sum(linear_scaled * [0, 1, 2])
        >>>
        >>> dataset_result = dataset.annotate_entries(dosage = get_dosage(dataset.PL))

        Note
        ----
        This method does not support aggregation.

        Notes
        -----
        This method creates new entry fields, but can also overwrite existing fields. Only
        same-scope fields can be overwritten: for example, it is not possible to annotate a
        global field `foo` and later create an entry field `foo`. However, it would be possible
        to create an entry field `foo` and later create another entry field `foo`, overwriting
        the first.

        The arguments to the method should either be :class:`.Expression`
        objects, or should be implicitly interpretable as expressions.

        Parameters
        ----------
        named_exprs : keyword args of :class:`.Expression`
            Field names and the expressions to compute them.

        Returns
        -------
        :class:`.MatrixTable`
            Matrix table with new row-and-column-indexed field(s).
        """
        caller = "MatrixTable.annotate_entries"
        e = get_annotate_exprs(caller, named_exprs, self._entry_indices)
        return self._select_entries(caller, s=self.entry.annotate(**e))

    def select_globals(self, *exprs, **named_exprs) -> 'MatrixTable':
        """Select existing global fields or create new fields by name, dropping the rest.

        Examples
        --------
        Select one existing field and compute a new one:

        >>> dataset_result = dataset.select_globals(dataset.global_field_1,
        ...                                         another_global=['AFR', 'EUR', 'EAS', 'AMR', 'SAS'])

        Notes
        -----
        This method creates new global fields. If a created field shares its name
        with a differently-indexed field of the table, the method will fail.

        Note
        ----

        See :meth:`.Table.select` for more information about using ``select`` methods.

        Note
        ----
        This method does not support aggregation.

        Parameters
        ----------
        exprs : variable-length args of :obj:`str` or :class:`.Expression`
            Arguments that specify field names or nested field reference expressions.
        named_exprs : keyword args of :class:`.Expression`
            Field names and the expressions to compute them.

        Returns
        -------
        :class:`.MatrixTable`
            MatrixTable with specified global fields.
        """

        exprs = [self[e] if not isinstance(e, Expression) else e for e in exprs]
        named_exprs = {k: to_expr(v) for k, v in named_exprs.items()}
        assignments = OrderedDict()

        for e in exprs:
            if not e._ir.is_nested_field:
                raise ExpressionException("method 'select_globals' expects keyword arguments for complex expressions")
            assert isinstance(e._ir, GetField)
            assignments[e._ir.name] = e

        for k, e in named_exprs.items():
            check_collisions(self._fields, k, self._global_indices)
            assignments[k] = e

        check_field_uniqueness(assignments.keys())
        return self._select_globals('MatrixTable.select_globals', hl.struct(**assignments))

    def select_rows(self, *exprs, **named_exprs) -> 'MatrixTable':
        """Select existing row fields or create new fields by name, dropping all
        other non-key fields.

        Examples
        --------
        Select existing fields and compute a new one:

        >>> dataset_result = dataset.select_rows(
        ...    dataset.variant_qc.gq_stats.mean,
        ...    high_quality_cases = agg.count_where((dataset.GQ > 20) &
        ...                                         dataset.is_case))

        Notes
        -----
        This method creates new row fields. If a created field shares its name
        with a differently-indexed field of the table, or with a row key, the
        method will fail.

        Row keys are preserved. To drop or change a row key field, use
        :meth:`MatrixTable.key_rows_by`.

        Note
        ----

        See :meth:`.Table.select` for more information about using ``select`` methods.

        Note
        ----
        This method supports aggregation over columns. For instance, the usage:

        >>> dataset_result = dataset.select_rows(mean_GQ = agg.mean(dataset.GQ))

        will compute the mean per row.

        Parameters
        ----------
        exprs : variable-length args of :obj:`str` or :class:`.Expression`
            Arguments that specify field names or nested field reference expressions.
        named_exprs : keyword args of :class:`.Expression`
            Field names and the expressions to compute them.

        Returns
        -------
        :class:`.MatrixTable`
            MatrixTable with specified row fields.
        """
        row = get_select_exprs("MatrixTable.select_rows",
                               exprs, named_exprs, self._row_indices,
                               protect_keys=True)
        return self._select_rows('MatrixTable.select_rows', self.row_key.annotate(**row))

    def select_cols(self, *exprs, **named_exprs) -> 'MatrixTable':
        """Select existing column fields or create new fields by name, dropping the rest.

        Examples
        --------
        Select existing fields and compute a new one:

        >>> dataset_result = dataset.select_cols(
        ...     dataset.sample_qc,
        ...     dataset.pheno.age,
        ...     isCohort1 = dataset.pheno.cohort_name == 'Cohort1')

        Notes
        -----
        This method creates new column fields. If a created field shares its name
        with a differently-indexed field of the table, the method will fail.

        Note
        ----

        See :meth:`.Table.select` for more information about using ``select`` methods.

        Note
        ----
        This method supports aggregation over rows. For instance, the usage:

        >>> dataset_result = dataset.select_cols(mean_GQ = agg.mean(dataset.GQ))

        will compute the mean per column.

        Parameters
        ----------
        exprs : variable-length args of :obj:`str` or :class:`.Expression`
            Arguments that specify field names or nested field reference expressions.
        named_exprs : keyword args of :class:`.Expression`
            Field names and the expressions to compute them.

        Returns
        -------
        :class:`.MatrixTable`
            MatrixTable with specified column fields.
        """
        col = get_select_exprs("MatrixTable.select_cols",
                               exprs, named_exprs, self._col_indices,
                               protect_keys=True)
        return self._select_cols('MatrixTable.select_cols', self.col_key.annotate(**col))

    def select_entries(self, *exprs, **named_exprs) -> 'MatrixTable':
        """Select existing entry fields or create new fields by name, dropping the rest.

        Examples
        --------
        Drop all entry fields aside from `GT`:

        >>> dataset_result = dataset.select_entries(dataset.GT)

        Notes
        -----
        This method creates new entry fields. If a created field shares its name
        with a differently-indexed field of the table, the method will fail.

        Note
        ----

        See :meth:`.Table.select` for more information about using ``select`` methods.

        Note
        ----
        This method does not support aggregation.

        Parameters
        ----------
        exprs : variable-length args of :obj:`str` or :class:`.Expression`
            Arguments that specify field names or nested field reference expressions.
        named_exprs : keyword args of :class:`.Expression`
            Field names and the expressions to compute them.

        Returns
        -------
        :class:`.MatrixTable`
            MatrixTable with specified entry fields.
        """
        entry = get_select_exprs("MatrixTable.select_entries",
                                 exprs, named_exprs, self._entry_indices,
                                 protect_keys=True)
        return self._select_entries("MatrixTable.select_entries", hl.struct(**entry))

    @typecheck_method(exprs=oneof(str, Expression))
    def drop(self, *exprs) -> 'MatrixTable':
        """Drop fields.

        Examples
        --------

        Drop fields `PL` (an entry field), `info` (a row field), and `pheno` (a column
        field): using strings:

        >>> dataset_result = dataset.drop('PL', 'info', 'pheno')

        Drop fields `PL` (an entry field), `info` (a row field), and `pheno` (a column
        field): using field references:

        >>> dataset_result = dataset.drop(dataset.PL, dataset.info, dataset.pheno)

        Drop a list of fields:

        >>> fields_to_drop = ['PL', 'info', 'pheno']
        >>> dataset_result = dataset.drop(*fields_to_drop)

        Notes
        -----

        This method can be used to drop global, row-indexed, column-indexed, or
        row-and-column-indexed (entry) fields. The arguments can be either strings
        (``'field'``), or top-level field references (``table.field`` or
        ``table['field']``).

        Key fields (belonging to either the row key or the column key) cannot be
        dropped using this method. In order to drop a key field, use :meth:`.key_rows_by`
        or :meth:`.key_cols_by` to remove the field from the key before dropping.

        While many operations exist independently for rows, columns, entries, and
        globals, only one is needed for dropping due to the lack of any necessary
        contextual information.

        Parameters
        ----------
        exprs : varargs of :obj:`str` or :class:`.Expression`
            Names of fields to drop or field reference expressions.

        Returns
        -------
        :class:`.MatrixTable`
            Matrix table without specified fields.
        """

        def check_key(name, keys):
            if name in keys:
                raise ValueError("MatrixTable.drop: cannot drop key field '{}'".format(name))
            return name

        all_field_exprs = {e: k for k, e in self._fields.items()}
        fields_to_drop = set()
        for e in exprs:
            if isinstance(e, Expression):
                if e in all_field_exprs:
                    fields_to_drop.add(all_field_exprs[e])
                else:
                    raise ExpressionException("method 'drop' expects string field names or top-level field expressions"
                                              " (e.g. 'foo', matrix.foo, or matrix['foo'])")
            else:
                assert isinstance(e, str)
                if e not in self._fields:
                    raise IndexError("matrix has no field '{}'".format(e))
                fields_to_drop.add(e)

        m = self
        global_fields = [field for field in fields_to_drop if self._fields[field]._indices == self._global_indices]
        if global_fields:
            m = m._select_globals("MatrixTable.drop", m.globals.drop(*global_fields))

        row_fields = [check_key(field, list(self.row_key)) for field in fields_to_drop if self._fields[field]._indices == self._row_indices]
        if row_fields:
            m = m._select_rows("MatrixTable.drop", row=m.row.drop(*row_fields))

        col_fields = [check_key(field, list(self.col_key)) for field in fields_to_drop if self._fields[field]._indices == self._col_indices]
        if col_fields:
            m = m._select_cols("MatrixTable.drop", m.col.drop(*col_fields))

        entry_fields = [field for field in fields_to_drop if self._fields[field]._indices == self._entry_indices]
        if entry_fields:
            m = m._select_entries("MatrixTable.drop", m.entry.drop(*entry_fields))

        return m

    @typecheck_method(expr=expr_bool, keep=bool)
    def filter_rows(self, expr, keep: bool = True) -> 'MatrixTable':
        """Filter rows of the matrix.

        Examples
        --------

        Keep rows where `variant_qc.AF` is below 1%:

        >>> dataset_result = dataset.filter_rows(dataset.variant_qc.AF[1] < 0.01, keep=True)

        Remove rows where `filters` is non-empty:

        >>> dataset_result = dataset.filter_rows(dataset.filters.size() > 0, keep=False)

        Notes
        -----

        The expression `expr` will be evaluated for every row of the table. If `keep`
        is ``True``, then rows where `expr` evaluates to ``False`` will be removed (the
        filter keeps the rows where the predicate evaluates to ``True``). If `keep` is
        ``False``, then rows where `expr` evaluates to ``False`` will be removed (the
        filter removes the rows where the predicate evaluates to ``True``).

        Warning
        -------
        When `expr` evaluates to missing, the row will be removed regardless of `keep`.

        Note
        ----
        This method supports aggregation over columns. For instance,

        >>> dataset_result = dataset.filter_rows(agg.mean(dataset.GQ) > 20.0)

        will remove rows where the mean GQ of all entries in the row is smaller than
        20.

        Parameters
        ----------
        expr : bool or :class:`.BooleanExpression`
            Filter expression.
        keep : bool
            Keep rows where `expr` is true.

        Returns
        -------
        :class:`.MatrixTable`
            Filtered matrix table.
        """
        caller = 'MatrixTable.filter_rows'
        analyze(caller, expr, self._row_indices, {self._col_axis})

        if expr._aggregations:
            bool_uid = Env.get_uid()
            mt = self._select_rows(caller, self.row.annotate(**{bool_uid: expr}))
            return mt.filter_rows(mt[bool_uid], keep).drop(bool_uid)

        base, cleanup = self._process_joins(expr)
        mt = MatrixTable(MatrixFilterRows(base._mir, filter_predicate_with_keep(expr._ir, keep)))
        return cleanup(mt)

    @typecheck_method(expr=expr_bool, keep=bool)
    def filter_cols(self, expr, keep: bool = True) -> 'MatrixTable':
        """Filter columns of the matrix.

        Examples
        --------

        Keep columns where `pheno.is_case` is ``True`` and `pheno.age` is larger
        than 50:

        >>> dataset_result = dataset.filter_cols(dataset.pheno.is_case &
        ...                                      (dataset.pheno.age > 50),
        ...                                      keep=True)

        Remove columns where `sample_qc.gq_stats.mean` is less than 20:

        >>> dataset_result = dataset.filter_cols(dataset.sample_qc.gq_stats.mean < 20,
        ...                                      keep=False)

        Remove columns where `s` is found in a Python set:

        >>> samples_to_remove = {'NA12878', 'NA12891', 'NA12892'}
        >>> set_to_remove = hl.literal(samples_to_remove)
        >>> dataset_result = dataset.filter_cols(~set_to_remove.contains(dataset['s']))

        Notes
        -----

        The expression `expr` will be evaluated for every column of the table. If
        `keep` is ``True``, then columns where `expr` evaluates to ``False`` will be
        removed (the filter keeps the columns where the predicate evaluates to
        ``True``). If `keep` is ``False``, then columns where `expr` evaluates to
        ``False`` will be removed (the filter removes the columns where the predicate
        evaluates to ``True``).

        Warning
        -------
        When `expr` evaluates to missing, the column will be removed regardless of
        `keep`.

        Note
        ----
        This method supports aggregation over rows. For instance,

        >>> dataset_result = dataset.filter_cols(agg.mean(dataset.GQ) > 20.0)

        will remove columns where the mean GQ of all entries in the column is smaller
        than 20.

        Parameters
        ----------
        expr : bool or :class:`.BooleanExpression`
            Filter expression.
        keep : bool
            Keep columns where `expr` is true.

        Returns
        -------
        :class:`.MatrixTable`
            Filtered matrix table.
        """
        caller = 'MatrixTable.filter_cols'
        analyze(caller, expr, self._col_indices, {self._row_axis})

        if expr._aggregations:
            bool_uid = Env.get_uid()
            mt = self._select_cols(caller, self.col.annotate(**{bool_uid: expr}))
            return mt.filter_cols(mt[bool_uid], keep).drop(bool_uid)

        base, cleanup = self._process_joins(expr)
        mt = MatrixTable(MatrixFilterCols(base._mir, filter_predicate_with_keep(expr._ir, keep)))
        return cleanup(mt)

    @typecheck_method(expr=expr_bool, keep=bool)
    def filter_entries(self, expr, keep: bool = True) -> 'MatrixTable':
        """Filter entries of the matrix.

        Examples
        --------

        Keep entries where the sum of `AD` is greater than 10 and `GQ` is greater than 20:

        >>> dataset_result = dataset.filter_entries((hl.sum(dataset.AD) > 10) & (dataset.GQ > 20))

        Notes
        -----

        The expression `expr` will be evaluated for every entry of the table. If
        `keep` is ``True``, then entries where `expr` evaluates to ``False`` will be
        removed (the filter keeps the entries where the predicate evaluates to
        ``True``). If `keep` is ``False``, then entries where `expr` evaluates to
        ``False`` will be removed (the filter removes the entries where the predicate
        evaluates to ``True``).

        Note
        ----
        "Removal" of an entry constitutes setting all its fields to missing. There
        is some debate about what removing an entry of a matrix means semantically,
        given the representation of a :class:`.MatrixTable` as a whole workspace in
        Hail.

        Warning
        -------
        When `expr` evaluates to missing, the entry will be removed regardless of
        `keep`.

        Note
        ----
        This method does not support aggregation.

        Parameters
        ----------
        expr : bool or :class:`.BooleanExpression`
            Filter expression.
        keep : bool
            Keep entries where `expr` is true.

        Returns
        -------
        :class:`.MatrixTable`
            Filtered matrix table.
        """
        base, cleanup = self._process_joins(expr)
        analyze('MatrixTable.filter_entries', expr, self._entry_indices)

        m = MatrixTable(MatrixFilterEntries(base._mir, filter_predicate_with_keep(expr._ir, keep)))
        return cleanup(m)

    def transmute_globals(self, **named_exprs) -> 'MatrixTable':
        """Similar to :meth:`.MatrixTable.annotate_globals`, but drops referenced fields.

        Notes
        -----
        This method adds new global fields according to `named_exprs`, and
        drops all global fields referenced in those expressions. See
        :meth:`.Table.transmute` for full documentation on how transmute
        methods work.

        See Also
        --------
        :meth:`.Table.transmute`, :meth:`.MatrixTable.select_globals`,
        :meth:`.MatrixTable.annotate_globals`

        Parameters
        ----------
        named_exprs : keyword args of :class:`.Expression`
            Annotation expressions.

        Returns
        -------
        :class:`.MatrixTable`
        """
        caller = 'MatrixTable.transmute_globals'
        e = get_annotate_exprs(caller, named_exprs, self._global_indices)
        fields_referenced = extract_refs_by_indices(e.values(), self._global_indices) - set(e.keys())

        return self._select_globals(caller,
                                    self.globals.annotate(**named_exprs).drop(*fields_referenced))

    def transmute_rows(self, **named_exprs) -> 'MatrixTable':
        """Similar to :meth:`.MatrixTable.annotate_rows`, but drops referenced fields.

        Notes
        -----
        This method adds new row fields according to `named_exprs`, and drops
        all row fields referenced in those expressions. See
        :meth:`.Table.transmute` for full documentation on how transmute
        methods work.

        Note
        ----
        :meth:`transmute_rows` will not drop key fields.

        Note
        ----
        This method supports aggregation over columns.

        See Also
        --------
        :meth:`.Table.transmute`, :meth:`.MatrixTable.select_rows`,
        :meth:`.MatrixTable.annotate_rows`

        Parameters
        ----------
        named_exprs : keyword args of :class:`.Expression`
            Annotation expressions.

        Returns
        -------
        :class:`.MatrixTable`
        """
        caller = 'MatrixTable.transmute_rows'
        e = get_annotate_exprs(caller, named_exprs, self._row_indices)
        fields_referenced = extract_refs_by_indices(e.values(), self._row_indices) - set(e.keys())
        fields_referenced -= set(self.row_key)

        return self._select_rows(caller, self.row.annotate(**named_exprs).drop(*fields_referenced))

    def transmute_cols(self, **named_exprs) -> 'MatrixTable':
        """Similar to :meth:`.MatrixTable.annotate_cols`, but drops referenced fields.

        Notes
        -----
        This method adds new column fields according to `named_exprs`, and
        drops all column fields referenced in those expressions. See
        :meth:`.Table.transmute` for full documentation on how transmute
        methods work.

        Note
        ----
        :meth:`transmute_cols` will not drop key fields.

        Note
        ----
        This method supports aggregation over rows.

        See Also
        --------
        :meth:`.Table.transmute`, :meth:`.MatrixTable.select_cols`,
        :meth:`.MatrixTable.annotate_cols`

        Parameters
        ----------
        named_exprs : keyword args of :class:`.Expression`
            Annotation expressions.

        Returns
        -------
        :class:`.MatrixTable`
        """
        caller = 'MatrixTable.transmute_cols'
        e = get_annotate_exprs(caller, named_exprs, self._col_indices)
        fields_referenced = extract_refs_by_indices(e.values(), self._col_indices) - set(e.keys())
        fields_referenced -= set(self.col_key)

        return self._select_cols(caller,
                                 self.col.annotate(**named_exprs).drop(*fields_referenced))

    def transmute_entries(self, **named_exprs):
        """Similar to :meth:`.MatrixTable.annotate_entries`, but drops referenced fields.

        Notes
        -----
        This method adds new entry fields according to `named_exprs`, and
        drops all entry fields referenced in those expressions. See
        :meth:`.Table.transmute` for full documentation on how transmute
        methods work.

        See Also
        --------
        :meth:`.Table.transmute`, :meth:`.MatrixTable.select_entries`,
        :meth:`.MatrixTable.annotate_entries`

        Parameters
        ----------
        named_exprs : keyword args of :class:`.Expression`
            Annotation expressions.

        Returns
        -------
        :class:`.MatrixTable`
        """
        caller = 'MatrixTable.transmute_entries'
        e = get_annotate_exprs(caller, named_exprs, self._entry_indices)
        fields_referenced = extract_refs_by_indices(e.values(), self._entry_indices) - set(e.keys())

        return self._select_entries(caller,
                                    self.entry.annotate(**named_exprs).drop(*fields_referenced))

    @typecheck_method(expr=expr_any)
    def aggregate_rows(self, expr) -> Any:
        """Aggregate over rows to a local value.

        Examples
        --------
        Aggregate over rows:

        >>> dataset.aggregate_rows(hl.struct(n_high_quality=agg.count_where(dataset.qual > 40),
        ...                                  mean_qual=agg.mean(dataset.qual)))
        Struct(n_high_quality=13, mean_qual=544323.8915384616)

        Notes
        -----
        Unlike most :class:`.MatrixTable` methods, this method does not support
        meaningful references to fields that are not global or indexed by row.

        This method should be thought of as a more convenient alternative to
        the following:

        >>> rows_table = dataset.rows()
        >>> rows_table.aggregate(hl.struct(n_high_quality=agg.count_where(rows_table.qual > 40),
        ...                                mean_qual=agg.mean(rows_table.qual)))

        Note
        ----
        This method supports (and expects!) aggregation over rows.

        Parameters
        ----------
        expr : :class:`.Expression`
            Aggregation expression.

        Returns
        -------
        any
            Aggregated value dependent on `expr`.
        """
        base, _ = self._process_joins(expr)

        analyze('MatrixTable.aggregate_rows', expr, self._global_indices, {self._row_axis})

        result_json = base._jmt.aggregateRowsJSON(str(expr._ir))
        return expr.dtype._from_json(result_json)

    @typecheck_method(expr=expr_any)
    def aggregate_cols(self, expr) -> Any:
        """Aggregate over columns to a local value.

        Examples
        --------
        Aggregate over columns:

        >>> dataset.aggregate_cols(
        ...    hl.struct(fraction_female=agg.fraction(dataset.pheno.is_female),
        ...              case_ratio=agg.count_where(dataset.is_case) / agg.count()))
        Struct(fraction_female=0.48, case_ratio=1.0)

        Notes
        -----
        Unlike most :class:`.MatrixTable` methods, this method does not support
        meaningful references to fields that are not global or indexed by column.

        This method should be thought of as a more convenient alternative to
        the following:

        >>> cols_table = dataset.cols()
        >>> cols_table.aggregate(
        ...     hl.struct(fraction_female=agg.fraction(cols_table.pheno.is_female),
        ...               case_ratio=agg.count_where(cols_table.is_case) / agg.count()))

        Note
        ----
        This method supports (and expects!) aggregation over columns.

        Parameters
        ----------
        expr : :class:`.Expression`
            Aggregation expression.

        Returns
        -------
        any
            Aggregated value dependent on `expr`.
        """
        base, _ = self._process_joins(expr)

        analyze('MatrixTable.aggregate_cols', expr, self._global_indices, {self._col_axis})

        result_json = base._jmt.aggregateColsJSON(str(expr._ir))
        return expr.dtype._from_json(result_json)

    @typecheck_method(expr=expr_any)
    def aggregate_entries(self, expr) -> Any:
        """Aggregate over entries to a local value.

        Examples
        --------
        Aggregate over entries:

        >>> dataset.aggregate_entries(hl.struct(global_gq_mean=agg.mean(dataset.GQ),
        ...                                     call_rate=agg.fraction(hl.is_defined(dataset.GT))))
        Struct(global_gq_mean=64.01841473178543, call_rate=0.9607692307692308)

        Notes
        -----
        This method should be thought of as a more convenient alternative to
        the following:

        >>> entries_table = dataset.entries()
        >>> entries_table.aggregate(hl.struct(global_gq_mean=agg.mean(entries_table.GQ),
        ...                                   call_rate=agg.fraction(hl.is_defined(entries_table.GT))))

        Note
        ----
        This method supports (and expects!) aggregation over entries.

        Parameters
        ----------
        expr : :class:`.Expression`
            Aggregation expressions.

        Returns
        -------
        any
            Aggregated value dependent on `expr`.
        """

        base, _ = self._process_joins(expr)

        analyze('MatrixTable.aggregate_entries', expr, self._global_indices, {self._row_axis, self._col_axis})

        result_json = base._jmt.aggregateEntriesJSON(str(expr._ir))
        return expr.dtype._from_json(result_json)

    @typecheck_method(field_expr=oneof(str, Expression))
    def explode_rows(self, field_expr) -> 'MatrixTable':
        """Explodes a row field of type array or set, copying the entire row for each element.

        Examples
        --------
        Explode rows by annotated genes:

        >>> dataset_result = dataset.explode_rows(dataset.gene)

        Notes
        -----
        The new matrix table will have `N` copies of each row, where `N` is the number
        of elements that row contains for the field denoted by `field_expr`. The field
        referenced in `field_expr` is replaced in the sequence of duplicated rows by the
        sequence of elements in the array or set. All other fields remain the same,
        including entry fields.

        If the field referenced with `field_expr` is missing or empty, the row is
        removed entirely.

        Parameters
        ----------
        field_expr : str or :class:`.Expression`
            Field name or (possibly nested) field reference expression.

        Returns
        -------
        :class:MatrixTable`
            Matrix table exploded row-wise for each element of `field_expr`.
        """
        if isinstance(field_expr, str):
            if not field_expr in self._fields:
                raise KeyError("MatrixTable has no field '{}'".format(field_expr))
            elif self._fields[field_expr]._indices != self._row_indices:
                raise ExpressionException("Method 'explode_rows' expects a field indexed by row, found axes '{}'"
                                          .format(self._fields[field_expr]._indices.axes))
            root = [field_expr]
            field_expr = self._fields[field_expr]
        else:
            analyze('MatrixTable.explode_rows', field_expr, self._row_indices, set(self._fields.keys()))
            if not field_expr._ir.is_nested_field:
                raise ExpressionException(
                    "method 'explode_rows' requires a field or subfield, not a complex expression")
            nested = field_expr._ir
            root = []
            while isinstance(nested, GetField):
                root.append(nested.name)
                nested = nested.o
            root = [r for r in reversed(root)]

        if not isinstance(field_expr.dtype, (tarray, tset)):
            raise ValueError(f"method 'explode_rows' expects array or set, found: {field_expr.dtype}")

        if self.row_key is not None:
            for k in self.row_key.values():
                if k is field_expr:
                    raise ValueError(f"method 'explode_rows' cannot explode a key field")

        return MatrixTable(MatrixExplodeRows(self._mir, root))

    @typecheck_method(field_expr=oneof(str, Expression))
    def explode_cols(self, field_expr) -> 'MatrixTable':
        """Explodes a column field of type array or set, copying the entire column for each element.

        Examples
        --------
        Explode columns by annotated cohorts:

        >>> dataset_result = dataset.explode_cols(dataset.cohorts)

        Notes
        -----
        The new matrix table will have `N` copies of each column, where `N` is the
        number of elements that column contains for the field denoted by `field_expr`.
        The field referenced in `field_expr` is replaced in the sequence of duplicated
        columns by the sequence of elements in the array or set. All other fields remain
        the same, including entry fields.

        If the field referenced with `field_expr` is missing or empty, the column is
        removed entirely.

        Parameters
        ----------
        field_expr : str or :class:`.Expression`
            Field name or (possibly nested) field reference expression.

        Returns
        -------
        :class:`.MatrixTable`
            Matrix table exploded column-wise for each element of `field_expr`.
        """

        if isinstance(field_expr, str):
            if not field_expr in self._fields:
                raise KeyError("MatrixTable has no field '{}'".format(field_expr))
            elif self._fields[field_expr]._indices != self._col_indices:
                raise ExpressionException("Method 'explode_cols' expects a field indexed by col, found axes '{}'"
                                          .format(self._fields[field_expr]._indices.axes))
            root = [field_expr]
            field_expr = self._fields[field_expr]
        else:
            analyze('MatrixTable.explode_cols', field_expr, self._col_indices)
            if not field_expr._ir.is_nested_field:
                raise ExpressionException(
                    "method 'explode_cols' requires a field or subfield, not a complex expression")
            root = []
            nested = field_expr._ir
            while isinstance(nested, GetField):
                root.append(nested.name)
                nested = nested.o
            root = [r for r in reversed(root)]

        if not isinstance(field_expr.dtype, (tarray, tset)):
            raise ValueError(f"method 'explode_cols' expects array or set, found: {field_expr.dtype}")

        if self.col_key is not None:
            for k in self.col_key.values():
                if k is field_expr:
                    raise ValueError(f"method 'explode_cols' cannot explode a key field")

        return MatrixTable(MatrixExplodeCols(self._mir, root))

    @typecheck_method(exprs=oneof(str, Expression), named_exprs=expr_any)
    def group_rows_by(self, *exprs, **named_exprs) -> 'GroupedMatrixTable':
        """Group rows, used with :meth:`.GroupedMatrixTable.aggregate`.

        Examples
        --------
        Aggregate to a matrix with genes as row keys, computing the number of
        non-reference calls as an entry field:

        >>> dataset_result = (dataset.group_rows_by(dataset.gene)
        ...                          .aggregate(n_non_ref = agg.count_where(dataset.GT.is_non_ref())))

        Notes
        -----
        All complex expressions must be passed as named expressions.

        Parameters
        ----------
        exprs : args of :obj:`str` or :class:`.Expression`
            Row fields to group by.
        named_exprs : keyword args of :class:`.Expression`
            Row-indexed expressions to group by.

        Returns
        -------
        :class:`.GroupedMatrixTable`
            Grouped matrix. Can be used to call :meth:`.GroupedMatrixTable.aggregate`.
        """

        return GroupedMatrixTable(self).group_rows_by(*exprs, **named_exprs)

    @typecheck_method(exprs=oneof(str, Expression), named_exprs=expr_any)
    def group_cols_by(self, *exprs, **named_exprs) -> 'GroupedMatrixTable':
        """Group columns, used with :meth:`.GroupedMatrixTable.aggregate`.

        Examples
        --------
        Aggregate to a matrix with cohort as column keys, computing the call rate
        as an entry field:

        >>> dataset_result = (dataset.group_cols_by(dataset.cohort)
        ...                          .aggregate(call_rate = agg.fraction(hl.is_defined(dataset.GT))))

        Notes
        -----
        All complex expressions must be passed as named expressions.

        Parameters
        ----------
        exprs : args of :obj:`str` or :class:`.Expression`
            Column fields to group by.
        named_exprs : keyword args of :class:`.Expression`
            Column-indexed expressions to group by.

        Returns
        -------
        :class:`.GroupedMatrixTable`
            Grouped matrix, can be used to call :meth:`.GroupedMatrixTable.aggregate`.
        """
        new_keys = []
        for e in exprs:
            if isinstance(e, str):
                e = self[e]
            else:
                e = to_expr(e)
            analyze('MatrixTable.group_cols_by', e, self._col_indices)
            if not e._ir.is_nested_field:
                raise ExpressionException("method 'group_cols_by' expects keyword arguments for complex expressions")
            key = e._ir.name
            if key in new_keys:
                raise ExpressionException("method 'group_cols_by' found duplicate field: {}".format(key))
            new_keys.append(key)

        ds = self.annotate_cols(**named_exprs)
        for key in named_exprs.keys():
            if key in new_keys:
                raise ExpressionException("method 'group_cols_by' found duplicate field: {}".format(key))
            new_keys.append(key)

        return GroupedMatrixTable(self).group_cols_by(*exprs, **named_exprs)

    def collect_cols_by_key(self) -> 'MatrixTable':
        """Collect values for each unique column key into arrays.

        Examples
        --------
        >>> mt = hl.utils.range_matrix_table(3, 3)
        >>> col_dict = hl.literal({0: [1], 1: [2, 3], 2: [4, 5, 6]})
        >>> mt = (mt.annotate_cols(foo = col_dict.get(mt.col_idx))
        ...     .explode_cols('foo'))
        >>> mt = mt.annotate_entries(bar = mt.row_idx * mt.foo)

        >>> mt.cols().show()
        +---------+-------+
        | col_idx |   foo |
        +---------+-------+
        |   int32 | int32 |
        +---------+-------+
        |       0 |     1 |
        |       1 |     2 |
        |       1 |     3 |
        |       2 |     4 |
        |       2 |     5 |
        |       2 |     6 |
        +---------+-------+

        >>> mt.entries().show()
        +---------+---------+-------+-------+
        | row_idx | col_idx |   foo |   bar |
        +---------+---------+-------+-------+
        |   int32 |   int32 | int32 | int32 |
        +---------+---------+-------+-------+
        |       0 |       0 |     1 |     0 |
        |       0 |       1 |     2 |     0 |
        |       0 |       1 |     3 |     0 |
        |       0 |       2 |     4 |     0 |
        |       0 |       2 |     5 |     0 |
        |       0 |       2 |     6 |     0 |
        |       1 |       0 |     1 |     1 |
        |       1 |       1 |     2 |     2 |
        |       1 |       1 |     3 |     3 |
        |       1 |       2 |     4 |     4 |
        +---------+---------+-------+-------+
        showing top 10 rows

        >>> mt = mt.collect_cols_by_key()
        >>> mt.cols().show()
        +---------+--------------+
        | col_idx | foo          |
        +---------+--------------+
        |   int32 | array<int32> |
        +---------+--------------+
        |       0 | [1]          |
        |       1 | [2,3]        |
        |       2 | [4,5,6]      |
        +---------+--------------+

        >>> mt.entries().show()
        +---------+---------+--------------+--------------+
        | row_idx | col_idx | foo          | bar          |
        +---------+---------+--------------+--------------+
        |   int32 |   int32 | array<int32> | array<int32> |
        +---------+---------+--------------+--------------+
        |       0 |       0 | [1]          | [0]          |
        |       0 |       1 | [2,3]        | [0,0]        |
        |       0 |       2 | [4,5,6]      | [0,0,0]      |
        |       1 |       0 | [1]          | [1]          |
        |       1 |       1 | [2,3]        | [2,3]        |
        |       1 |       2 | [4,5,6]      | [4,5,6]      |
        |       2 |       0 | [1]          | [2]          |
        |       2 |       1 | [2,3]        | [4,6]        |
        |       2 |       2 | [4,5,6]      | [8,10,12]    |
        +---------+---------+--------------+--------------+

        Notes
        -----
        Each entry field and each non-key column field of type t is replaced by
        a field of type array<t>. The value of each such field is an array
        containing all values of that field sharing the corresponding column
        key. In each column, the newly collected arrays all have the same
        length, and the values of each pre-collection column are guaranteed to
        be located at the same index in their corresponding arrays.

        Note
        -----
        The order of the columns is not guaranteed.

        Returns
        -------
        :class:`.MatrixTable`
        """

        return MatrixTable(MatrixCollectColsByKey(self._mir))

    def count_rows(self) -> int:
        """Count the number of rows in the matrix.

        Examples
        --------

        Count the number of rows:

        >>> n_rows = dataset.count_rows()

        Returns
        -------
        :obj:`int`
            Number of rows in the matrix.
        """

        return self._jmt.countRows()

    def _force_count_rows(self):
        return self._jmt.forceCountRows()

    def _force_count_cols(self):
        return self._jmt.forceCountCols()

    def count_cols(self) -> int:
        """Count the number of columns in the matrix.

        Examples
        --------

        Count the number of columns:

        >>> n_cols = dataset.count_cols()

        Returns
        -------
        :obj:`int`
            Number of columns in the matrix.
        """
        return self._jmt.countCols()

    def count(self) -> Tuple[int, int]:
        """Count the number of rows and columns in the matrix.

        Examples
        --------

        >>> dataset.count()

        Returns
        -------
        :obj:`int`, :obj:`int`
            Number of rows, number of cols.
        """
        r = self._jmt.count()
        return r._1(), r._2()

    @typecheck_method(output=str,
                      overwrite=bool,
                      stage_locally=bool,
                      _codec_spec=nullable(str))
    def write(self, output: str, overwrite: bool = False, stage_locally: bool = False,
              _codec_spec: Optional[str] = None):
        """Write to disk.

        Examples
        --------

        >>> dataset.write('output/dataset.mt')

        Warning
        -------
        Do not write to a path that is being read from in the same computation.

        Parameters
        ----------
        output : str
            Path at which to write.
        stage_locally: bool
            If ``True``, major output will be written to temporary local storage
            before being copied to ``output``
        overwrite : bool
            If ``True``, overwrite an existing file at the destination.
        """

        self._jmt.write(output, overwrite, stage_locally, _codec_spec)

    def globals_table(self) -> Table:
        """Returns a table with a single row with the globals of the matrix table.

        Examples
        --------
        Extract the globals table:

        >>> globals_table = dataset.globals_table()

        Returns
        -------
        :class:`.Table`
            Table with the globals from the matrix, with a single row.
        """
        return Table._from_java(self._jmt.globalsTable())

    def rows(self) -> Table:
        """Returns a table with all row fields in the matrix.

        Examples
        --------
        Extract the row table:

        >>> rows_table = dataset.rows()

        Returns
        -------
        :class:`.Table`
            Table with all row fields from the matrix, with one row per row of the matrix.
        """

        return Table(MatrixRowsTable(self._mir))

    def cols(self) -> Table:
        """Returns a table with all column fields in the matrix.

        Examples
        --------
        Extract the column table:

        >>> cols_table = dataset.cols()

        Warning
        -------
        Matrix table columns are typically sorted by the order at import, and
        not necessarily by column key. Since tables are always sorted by key,
        the table which results from this command will have its rows sorted by
        the column key (which becomes the table key). To preserve the original
        column order as the table row order, first unkey the columns using
        :meth:`key_cols_by` with no arguments.

        Returns
        -------
        :class:`.Table`
            Table with all column fields from the matrix, with one row per column of the matrix.
        """

        if len(self.col_key) != 0 and Env.hc()._warn_cols_order:
            warn("cols(): Resulting column table is sorted by 'col_key'."
                 "\n    To preserve matrix table column order, "
                 "first unkey columns with 'key_cols_by()'")
            Env.hc()._warn_cols_order = False

        return Table(MatrixColsTable(self._mir))

    def entries(self) -> Table:
        """Returns a matrix in coordinate table form.

        Examples
        --------
        Extract the entry table:

        >>> entries_table = dataset.entries()

        Warning
        -------
        The table returned by this method should be used for aggregation or queries,
        but never exported or written to disk without extensive filtering and field
        selection -- the disk footprint of an entries_table could be 100x (or more!)
        larger than its parent matrix. This means that if you try to export the entries
        table of a 10 terabyte matrix, you could write a petabyte of data!

        Warning
        -------
        Matrix table columns are typically sorted by the order at import, and
        not necessarily by column key. Since tables are always sorted by key,
        the table which results from this command will have its rows sorted by
        the compound (row key, column key) which becomes the table key.
        To preserve the original row-major entry order as the table row order,
        first unkey the columns using :meth:`key_cols_by` with no arguments.

        Returns
        -------
        :class:`.Table`
            Table with all non-global fields from the matrix, with **one row per entry of the matrix**.
        """
        if Env.hc()._warn_entries_order and len(self.col_key) > 0:
            warn("entries(): Resulting entries table is sorted by '(row_key, col_key)'."
                 "\n    To preserve row-major matrix table order, "
                 "first unkey columns with 'key_cols_by()'")
            Env.hc()._warn_entries_order = False

        return Table(MatrixEntriesTable(self._mir))

    def index_globals(self) -> Expression:
        """Return this matrix table's global variables for use in another
        expression context.

        Examples
        --------
        >>> dataset1 = dataset.annotate_globals(pli={'SCN1A': 0.999, 'SONIC': 0.014})
        >>> pli_dict = dataset1.index_globals().pli
        >>> dataset_result = dataset2.annotate_rows(gene_pli = dataset2.gene.map(lambda x: pli_dict.get(x)))

        Returns
        -------
        :class:`.StructExpression`
        """

        uid = Env.get_uid()

        def joiner(obj):
            if isinstance(obj, MatrixTable):
                return MatrixTable._from_java(Env.jutils().joinGlobals(obj._jmt, self._jmt, uid))
            else:
                assert isinstance(obj, Table)
                return Table._from_java(Env.jutils().joinGlobals(obj._jt, self._jmt, uid))

        ir = Join(GetField(TopLevelReference('global'), uid),
                  [uid],
                  [],
                  joiner)
        return construct_expr(ir, self.globals.dtype)

    def index_rows(self, *exprs):
        """Expose the row values as if looked up in a dictionary, indexing
        with `exprs`.

        Examples
        --------
        >>> dataset_result = dataset.annotate_rows(qual = dataset2.index_rows(dataset.locus, dataset.alleles).qual)

        Or equivalently:
        >>> dataset_result = dataset.annotate_rows(qual = dataset2.index_rows(dataset.row_key).qual)

        Parameters
        ----------
        exprs : variable-length args of :class:`.Expression`
            Index expressions.

        Notes
        -----
        :meth:`index_rows(exprs)` is equivalent to ``rows().index(exprs)``
        or ``rows()[exprs]``.

        The type of the resulting struct is the same as the type of
        :meth:`.row_value`.

        Returns
        -------
        :class:`.StructExpression`
        """
        exprs = [to_expr(e) for e in exprs]
        indices, aggregations = unify_all(*exprs)
        src = indices.source

        if aggregations:
            raise ExpressionException('Cannot join using an aggregated field')
        uid = Env.get_uid()
        uids_to_delete = [uid]

        if src is None:
            raise ExpressionException('Cannot index with a scalar expression')

        if not types_match(self.row_key.values(), exprs):
            if (len(exprs) == 1
                    and isinstance(exprs[0], TupleExpression)
                    and types_match(self.row_key.values(), exprs[0])):
                return self.index_rows(*exprs[0])
            elif (len(exprs) == 1
                  and isinstance(exprs[0], StructExpression)
                  and types_match(self.row_key.values(), exprs[0].values())):
                return self.index_rows(*exprs[0].values())
            elif len(exprs) != len(self.row_key):
                raise ExpressionException(f'Key mismatch: matrix table has {len(self.row_key)} row key fields, '
                                          f'found {len(exprs)} index expressions')
            else:
                raise ExpressionException(
                    f"Key type mismatch: cannot index matrix table with given expressions:\n"
                    f"  MatrixTable row key: {', '.join(str(t) for t in self.row_key.dtype.values())}\n"
                    f"  Index expressions:   {', '.join(str(e.dtype) for e in exprs)}")

        if isinstance(src, Table):
            # join table with matrix.rows_table()
            right = self.rows()
            return right.index(*exprs)
        else:
            assert isinstance(src, MatrixTable)
            right = self

            # fast path
            is_row_key = len(exprs) == len(src.row_key) and all(
                exprs[i] is src._fields[list(src.row_key)[i]] for i in range(len(exprs)))

            if is_row_key:
                def joiner(left):
                    return MatrixTable._from_java(left._jmt.annotateRowsVDS(right._jmt, uid))
                schema = tstruct(**{f: t for f, t in self.row.dtype.items() if f not in self.row_key})
                ir = Join(GetField(TopLevelReference('va'), uid),
                          uids_to_delete,
                          exprs,
                          joiner)
                return construct_expr(ir, schema, indices, aggregations)
            else:
                return self.rows().index(*exprs)

    def index_cols(self, *exprs):
        """Expose the column values as if looked up in a dictionary, indexing
        with `exprs`.

        Examples
        --------
        >>> dataset_result = dataset.annotate_cols(pheno = dataset2.index_cols(dataset.s).pheno)

        Or equivalently:
        >>> dataset_result = dataset.annotate_cols(pheno = dataset2.index_cols(dataset.col_key).pheno)

        Parameters
        ----------
        exprs : variable-length args of :class:`.Expression`
            Index expressions.

        Notes
        -----
        :meth:`index_cols(exprs)` is equivalent to ``cols().index(exprs)``
        or ``cols()[exprs]``.

        The type of the resulting struct is the same as the type of
        :meth:`.col_value`.

        Returns
        -------
        :class:`.StructExpression`
        """
        return self.cols().index(*exprs)

    def index_entries(self, row_exprs, col_exprs):
        """Expose the entries as if looked up in a dictionary, indexing
        with `exprs`.

        Examples
        --------
        >>> dataset_result = dataset.annotate_entries(GQ2 = dataset2.index_entries(dataset.row_key, dataset.col_key).GQ)

        Or equivalently:
        >>> dataset_result = dataset.annotate_entries(GQ2 = dataset2[dataset.row_key, dataset.col_key].GQ)

        Parameters
        ----------
        row_exprs : tuple of :class:`.Expression`
            Row index expressions.
        col_exprs : tuple of :class:`.Expression`
            Column index expressions.

        Notes
        -----
        The type of the resulting struct is the same as the type of
        :meth:`.entry`.

        Note
        ----
        There is a shorthand syntax for :meth:`.MatrixTable.index_entries` using
        square brackets (the Python ``__getitem__`` syntax). This syntax is
        preferred.

        >>> dataset_result = dataset.annotate_entries(GQ2 = dataset2[dataset.row_key, dataset.col_key].GQ)

        Returns
        -------
        :class:`.StructExpression`
        """
        row_exprs = wrap_to_tuple(row_exprs)
        col_exprs = wrap_to_tuple(col_exprs)
        if len(row_exprs) == 0  or len(col_exprs) == 0:
            raise ValueError("'MatrixTable.index_entries:' 'row_exprs' and 'col_exprs' must not be empty")
        row_non_exprs = list(filter(lambda e: not isinstance(e, Expression), row_exprs))
        if row_non_exprs:
            raise TypeError(f"'MatrixTable.index_entries': row_exprs expects expressions, found {row_non_exprs}")
        col_non_exprs = list(filter(lambda e: not isinstance(e, Expression), col_exprs))
        if col_non_exprs:
            raise TypeError(f"'MatrixTable.index_entries': col_exprs expects expressions, found {col_non_exprs}")

        if not types_match(self.row_key.values(), row_exprs):
            if (len(row_exprs) == 1
                    and isinstance(row_exprs[0], TupleExpression)
                    and types_match(self.row_key.values(), row_exprs[0])):
                return self.index_entries(tuple(row_exprs[0]), col_exprs)
            elif (len(row_exprs) == 1
                  and isinstance(row_exprs[0], StructExpression)
                  and types_match(self.row_key.values(), row_exprs[0].values())):
                return self.index_entries(tuple(row_exprs[0].values()), col_exprs)
            elif len(row_exprs) != len(self.row_key):
                raise ExpressionException(f'Key mismatch: matrix table has {len(self.row_key)} row key fields, '
                                          f'found {len(row_exprs)} index expressions')
            else:
                raise ExpressionException(
                    f"Cannot index table with given expressions\n"
                    f"  MatrixTable row key: {', '.join(str(t) for t in self.row_key.dtype.values())}\n"
                    f"  Index expressions:   {', '.join(str(e.dtype) for e in row_exprs)}")

        if not types_match(self.col_key.values(), col_exprs):
            if (len(col_exprs) == 1
                    and isinstance(col_exprs[0], TupleExpression)
                    and types_match(self.col_key.values(), col_exprs[0])):
                return self.index_entries(row_exprs, tuple(col_exprs[0]))
            elif (len(col_exprs) == 1
                  and isinstance(col_exprs[0], StructExpression)
                  and types_match(self.col_key.values(), col_exprs[0].values())):
                return self.index_entries(row_exprs, tuple(col_exprs[0].values()))
            elif len(col_exprs) != len(self.col_key):
                raise ExpressionException(f'Key mismatch: matrix table has {len(self.col_key)} col key fields, '
                                          f'found {len(col_exprs)} index expressions.')
            else:
                raise ExpressionException(
                    f"Key type mismatch: cannot index matrix table with given expressions:\n"
                    f"  MatrixTable col key: {', '.join(str(t) for t in self.col_key.dtype.values())}\n"
                    f"  Index expressions:   {', '.join(str(e.dtype) for e in col_exprs)}")

        indices, aggregations = unify_all(*(row_exprs + col_exprs))
        src = indices.source
        if aggregations:
            raise ExpressionException('Cannot join using an aggregated field')

        uid = Env.get_uid()
        uids = [uid]

        if isinstance(src, Table):
            # join table with matrix.entries_table()
            return self.entries().index(*(row_exprs + col_exprs))
        else:
            assert isinstance(src, MatrixTable)
            row_uid = Env.get_uid()
            uids.append(row_uid)
            col_uid = Env.get_uid()
            uids.append(col_uid)

            def joiner(left: MatrixTable):
                localized = self._localize_entries(row_uid, col_uid)
                src_cols_indexed = self.add_col_index(col_uid).cols()
                src_cols_indexed = src_cols_indexed.annotate(**{col_uid: hl.int32(src_cols_indexed[col_uid])})
                left = left._annotate_all(row_exprs = {row_uid: localized.index(*row_exprs)[row_uid]},
                                          col_exprs = {col_uid: src_cols_indexed.index(*col_exprs)[col_uid]})
                return left.annotate_entries(**{uid: left[row_uid][left[col_uid]]})

            ir = Join(GetField(TopLevelReference('g'), uid),
                      uids,
                      [*row_exprs, *col_exprs],
                      joiner)
            return construct_expr(ir, self.entry.dtype, indices, aggregations)

    @typecheck_method(entries_field_name=str, cols_field_name=str)
    def _localize_entries(self, entries_field_name, cols_field_name):
        return Table._from_java(self._jmt.localizeEntries(entries_field_name, cols_field_name))

    @typecheck_method(row_exprs=dictof(str, expr_any),
                      col_exprs=dictof(str, expr_any),
                      entry_exprs=dictof(str, expr_any),
                      global_exprs=dictof(str, expr_any))
    def _annotate_all(self,
                      row_exprs={},
                      col_exprs={},
                      entry_exprs={},
                      global_exprs={},
                      ) -> 'MatrixTable':
        all_exprs = list(itertools.chain(row_exprs.values(),
                                         col_exprs.values(),
                                         entry_exprs.values(),
                                         global_exprs.values()))

        base, cleanup = self._process_joins(*all_exprs)
        mir = base._mir

        if row_exprs:
            row_struct = InsertFields(base.row._ir, [(n, e._ir) for (n, e) in row_exprs.items()])
            mir = MatrixMapRows(mir, row_struct)
        if col_exprs:
            col_struct = InsertFields(base.col._ir, [(n, e._ir) for (n, e) in col_exprs.items()])
            mir = MatrixMapCols(mir, col_struct, None)
        if entry_exprs:
            entry_struct = InsertFields(base.entry._ir, [(n, e._ir) for (n, e) in entry_exprs.items()])
            mir = MatrixMapEntries(mir, entry_struct)
        if global_exprs:
            globals_struct = InsertFields(base.globals._ir, [(n, e._ir) for (n, e) in global_exprs.items()])
            mir = MatrixMapGlobals(mir, globals_struct)

        return cleanup(MatrixTable(mir))

    @typecheck_method(row_exprs=dictof(str, expr_any),
                      row_key=nullable(sequenceof(str)),
                      col_exprs=dictof(str, expr_any),
                      col_key=nullable(sequenceof(str)),
                      entry_exprs=dictof(str, expr_any),
                      global_exprs=dictof(str, expr_any))
    def _select_all(self,
                    row_exprs={},
                    row_key=None,
                    col_exprs={},
                    col_key=None,
                    entry_exprs={},
                    global_exprs={},
                    ) -> 'MatrixTable':
        all_exprs = list(itertools.chain(row_exprs.values(),
                                         col_exprs.values(),
                                         entry_exprs.values(),
                                         global_exprs.values()))

        base, cleanup = self._process_joins(*all_exprs)
        mir = base._mir

        if row_key is not None:
            mir = MatrixKeyRowsBy(mir, [])
        row_struct = hl.struct(**row_exprs)
        analyze("MatrixTable.select_rows", row_struct, self._row_indices)
        mir = MatrixMapRows(mir, row_struct._ir)
        if row_key is not None:
            mir = MatrixKeyRowsBy(mir, row_key)

        col_struct = hl.struct(**col_exprs)
        analyze("MatrixTable.select_cols", col_struct, self._col_indices)
        mir = MatrixMapCols(mir, col_struct._ir, col_key)

        entry_struct = hl.struct(**entry_exprs)
        analyze("MatrixTable.select_entries", entry_struct, self._entry_indices)
        mir = MatrixMapEntries(mir, entry_struct._ir)

        globals_struct = hl.struct(**global_exprs)
        analyze("MatrixTable.select_globals", globals_struct, self._global_indices)
        mir = MatrixMapGlobals(mir, globals_struct._ir)

        return cleanup(MatrixTable(mir))

    def _process_joins(self, *exprs):
        return process_joins(self, exprs)

    def describe(self, handler=print):
        """Print information about the fields in the matrix."""

        def format_type(typ):
            return typ.pretty(indent=4)

        if len(self.globals.dtype) == 0:
            global_fields = '\n    None'
        else:
            global_fields = ''.join("\n    '{name}': {type} ".format(
                name=f, type=format_type(t)) for f, t in self.globals.dtype.items())

        if len(self.row) == 0:
            row_fields = '\n    None'
        else:
            row_fields = ''.join("\n    '{name}': {type} ".format(
                name=f, type=format_type(t)) for f, t in self.row.dtype.items())

        row_key = '[' + ', '.join("'{name}'".format(name=f) for f in self.row_key) + ']' \
            if self.row_key else None

        if len(self.col) == 0:
            col_fields = '\n    None'
        else:
            col_fields = ''.join("\n    '{name}': {type} ".format(
                name=f, type=format_type(t)) for f, t in self.col.dtype.items())

        col_key = '[' + ', '.join("'{name}'".format(name=f) for f in self.col_key) + ']' \
            if self.col_key else None

        if len(self.entry) == 0:
            entry_fields = '\n    None'
        else:
            entry_fields = ''.join("\n    '{name}': {type} ".format(
                name=f, type=format_type(t)) for f, t in self.entry.dtype.items())

        s = '----------------------------------------\n' \
            'Global fields:{g}\n' \
            '----------------------------------------\n' \
            'Column fields:{c}\n' \
            '----------------------------------------\n' \
            'Row fields:{r}\n' \
            '----------------------------------------\n' \
            'Entry fields:{e}\n' \
            '----------------------------------------\n' \
            'Column key: {ck}\n' \
            'Row key: {rk}\n' \
            '----------------------------------------'.format(g=global_fields,
                                                              rk=row_key,
                                                              r=row_fields,
                                                              ck=col_key,
                                                              c=col_fields,
                                                              e=entry_fields)
        handler(s)

    @typecheck_method(indices=sequenceof(int))
    def choose_cols(self, indices: List[int]) -> 'MatrixTable':
        """Choose a new set of columns from a list of old column indices.

        Examples
        --------

        Randomly shuffle column order:

        >>> import random
        >>> indices = list(range(dataset.count_cols()))
        >>> random.shuffle(indices)
        >>> dataset_reordered = dataset.choose_cols(indices)

        Take the first ten columns:

        >>> dataset_result = dataset.choose_cols(list(range(10)))

        Parameters
        ----------
        indices : :obj:`list` of :obj:`int`
            List of old column indices.

        Returns
        -------
        :class:`.MatrixTable`
        """
        n_cols = self.count_cols()
        for i in indices:
            if not 0 <= i < n_cols:
                raise ValueError(f"'choose_cols': expect indices between 0 and {n_cols}, found {i}")
        return MatrixTable(MatrixChooseCols(self._mir, indices))

    def n_partitions(self) -> int:
        """Number of partitions.

        Notes
        -----

        The data in a dataset is divided into chunks called partitions, which
        may be stored together or across a network, so that each partition may
        be read and processed in parallel by available cores. Partitions are a
        core concept of distributed computation in Spark, see `here
        <http://spark.apache.org/docs/latest/programming-guide.html#resilient-distributed-datasets-rdds>`__
        for details.

        Returns
        -------
        int
            Number of partitions.
        """
        return self._jmt.nPartitions()

    @typecheck_method(n_partitions=int,
                      shuffle=bool)
    def repartition(self, n_partitions: int, shuffle: bool = True) -> 'MatrixTable':
        """Increase or decrease the number of partitions.

        Examples
        --------

        Repartition to 500 partitions:

        >>> dataset_result = dataset.repartition(500)

        Notes
        -----

        Check the current number of partitions with :meth:`.n_partitions`.

        The data in a dataset is divided into chunks called partitions, which
        may be stored together or across a network, so that each partition may
        be read and processed in parallel by available cores. When a matrix with
        :math:`M` rows is first imported, each of the :math:`k` partitions will
        contain about :math:`M/k` of the rows. Since each partition has some
        computational overhead, decreasing the number of partitions can improve
        performance after significant filtering. Since it's recommended to have
        at least 2 - 4 partitions per core, increasing the number of partitions
        can allow one to take advantage of more cores. Partitions are a core
        concept of distributed computation in Spark, see `their documentation
        <http://spark.apache.org/docs/latest/programming-guide.html#resilient-distributed-datasets-rdds>`__
        for details. With ``shuffle=True``, Hail does a full shuffle of the data
        and creates equal sized partitions. With ``shuffle=False``, Hail
        combines existing partitions to avoid a full shuffle. These algorithms
        correspond to the `repartition` and `coalesce` commands in Spark,
        respectively. In particular, when ``shuffle=False``, ``n_partitions``
        cannot exceed current number of partitions.

        Note
        ----
        If `shuffle` is ``False``, the number of partitions may only be
        reduced, not increased.

        Parameters
        ----------
        n_partitions : int
            Desired number of partitions.
        shuffle : bool
            If ``True``, use full shuffle to repartition.

        Returns
        -------
        :class:`.MatrixTable`
            Repartitioned dataset.
        """
        return MatrixTable(MatrixRepartition(self._mir, n_partitions, shuffle))

    @typecheck_method(max_partitions=int)
    def naive_coalesce(self, max_partitions: int) -> 'MatrixTable':
        """Naively decrease the number of partitions.

        Example
        -------
        Naively repartition to 10 partitions:

        >>> dataset_result = dataset.naive_coalesce(10)

        Warning
        -------
        :meth:`.naive_coalesce` simply combines adjacent partitions to achieve
        the desired number. It does not attempt to rebalance, unlike
        :meth:`.repartition`, so it can produce a heavily unbalanced dataset. An
        unbalanced dataset can be inefficient to operate on because the work is
        not evenly distributed across partitions.

        Parameters
        ----------
        max_partitions : int
            Desired number of partitions. If the current number of partitions is
            less than or equal to `max_partitions`, do nothing.

        Returns
        -------
        :class:`.MatrixTable`
            Matrix table with at most `max_partitions` partitions.
        """
        return MatrixTable._from_java(self._jmt.naiveCoalesce(max_partitions))

    def cache(self) -> 'MatrixTable':
        """Persist the dataset in memory.

        Examples
        --------
        Persist the dataset in memory:

        >>> dataset = dataset.cache() # doctest: +SKIP

        Notes
        -----

        This method is an alias for :func:`persist("MEMORY_ONLY") <hail.MatrixTable.persist>`.

        Returns
        -------
        :class:`.MatrixTable`
            Cached dataset.
        """
        return self.persist('MEMORY_ONLY')

    @typecheck_method(storage_level=storage_level)
    def persist(self, storage_level: str = 'MEMORY_AND_DISK') -> 'MatrixTable':
        """Persist this table in memory or on disk.

        Examples
        --------
        Persist the dataset to both memory and disk:

        >>> dataset = dataset.persist() # doctest: +SKIP

        Notes
        -----

        The :meth:`.MatrixTable.persist` and :meth:`.MatrixTable.cache`
        methods store the current dataset on disk or in memory temporarily to
        avoid redundant computation and improve the performance of Hail
        pipelines. This method is not a substitution for :meth:`.Table.write`,
        which stores a permanent file.

        Most users should use the "MEMORY_AND_DISK" storage level. See the `Spark
        documentation
        <http://spark.apache.org/docs/latest/programming-guide.html#rdd-persistence>`__
        for a more in-depth discussion of persisting data.

        Parameters
        ----------
        storage_level : str
            Storage level.  One of: NONE, DISK_ONLY,
            DISK_ONLY_2, MEMORY_ONLY, MEMORY_ONLY_2, MEMORY_ONLY_SER,
            MEMORY_ONLY_SER_2, MEMORY_AND_DISK, MEMORY_AND_DISK_2,
            MEMORY_AND_DISK_SER, MEMORY_AND_DISK_SER_2, OFF_HEAP

        Returns
        -------
        :class:`.MatrixTable`
            Persisted dataset.
        """
        return MatrixTable._from_java(self._jmt.persist(storage_level))

    def unpersist(self) -> 'MatrixTable':
        """
        Unpersists this dataset from memory/disk.

        Notes
        -----
        This function will have no effect on a dataset that was not previously
        persisted.

        Returns
        -------
        :class:`.MatrixTable`
            Unpersisted dataset.
        """
        return MatrixTable._from_java(self._jmt.unpersist())

    @typecheck_method(name=str)
    def add_row_index(self, name: str = 'row_idx') -> 'MatrixTable':
        """Add the integer index of each row as a new row field.

        Examples
        --------

        >>> dataset_result = dataset.add_row_index()

        Notes
        -----
        The field added is type :py:data:`.tint64`.

        The row index is 0-indexed; the values are found in the range
        ``[0, N)``, where ``N`` is the total number of rows.

        Parameters
        ----------
        name : :obj:`str`
            Name for row index field.

        Returns
        -------
        :class:`.MatrixTable`
            Dataset with new field.
        """
        return self.annotate_rows(**{name: hl.scan.count()})

    @typecheck_method(name=str)
    def add_col_index(self, name: str = 'col_idx') -> 'MatrixTable':
        """Add the integer index of each column as a new column field.

        Examples
        --------

        >>> dataset_result = dataset.add_col_index()

        Notes
        -----
        The field added is type :py:data:`.tint32`.

        The column index is 0-indexed; the values are found in the range
        ``[0, N)``, where ``N`` is the total number of columns.

        Parameters
        ----------
        name: :obj:`str`
            Name for column index field.

        Returns
        -------
        :class:`.MatrixTable`
            Dataset with new field.
        """
        return self.annotate_cols(**{name: hl.scan.count()})

    @typecheck_method(other=matrix_table_type,
                      tolerance=numeric,
                      absolute=bool)
    def _same(self, other, tolerance=1e-6, absolute=False):
        return self._jmt.same(other._jmt, tolerance, absolute)

    @typecheck_method(caller=str, s=expr_struct())
    def _select_entries(self, caller, s) -> 'MatrixTable':
        base, cleanup = self._process_joins(s)
        analyze(caller, s, self._entry_indices)
        return cleanup(MatrixTable(MatrixMapEntries(base._mir, s._ir)))

    @typecheck_method(caller=str,
                      row=expr_struct())
    def _select_rows(self, caller, row):
        analyze(caller, row, self._row_indices, {self._col_axis})
        base, cleanup = self._process_joins(row)
        return cleanup(MatrixTable(MatrixMapRows(base._mir, row._ir)))

    @typecheck_method(key_struct=expr_struct())
    def _select_rows_processed(self, key_struct):
        new_key = list(key_struct.keys())
        keys = Env.get_uid()
        fields = [(n, GetField(Ref(keys), n)) for (n, t) in key_struct.dtype.items()]
        row_ir = Let(keys, key_struct._ir, InsertFields(self.row._ir, fields))
        return MatrixTable(
            MatrixKeyRowsBy(
                MatrixMapRows(
                    MatrixKeyRowsBy(self._mir, []),
                row_ir),
            new_key))

    @typecheck_method(caller=str,
                      col=expr_struct(),
                      new_key=nullable(sequenceof(str)))
    def _select_cols(self, caller, col, new_key=None):
        analyze(caller, col, self._col_indices, {self._row_axis})
        base, cleanup = self._process_joins(col)
        return cleanup(MatrixTable(MatrixMapCols(base._mir, col._ir, new_key)))

    @typecheck_method(key_struct=expr_struct())
    def _select_cols_processed(self, key_struct):
        new_key = list(key_struct.keys())
        keys = Env.get_uid()
        fields = [(n, GetField(Ref(keys), n)) for (n, t) in key_struct.dtype.items()]
        col_ir = Let(keys, key_struct._ir, InsertFields(self.col._ir, fields))
        return MatrixTable(MatrixMapCols(self._mir, col_ir, new_key))

    @typecheck_method(caller=str, s=expr_struct())
    def _select_globals(self, caller, s) -> 'MatrixTable':
        base, cleanup = self._process_joins(s)
        analyze(caller, s, self._global_indices)
        return cleanup(MatrixTable(MatrixMapGlobals(base._mir, s._ir)))

    @typecheck(datasets=matrix_table_type)
    def union_rows(*datasets: 'MatrixTable') -> 'MatrixTable':
        """Take the union of dataset rows.

        Examples
        --------

        .. testsetup::

            dataset_to_union_1 = dataset
            dataset_to_union_2 = dataset

        Union the rows of two datasets:

        >>> dataset_result = dataset_to_union_1.union_rows(dataset_to_union_2)

        Given a list of datasets, take the union of all rows:

        >>> all_datasets = [dataset_to_union_1, dataset_to_union_2]

        The following three syntaxes are equivalent:

        >>> dataset_result = dataset_to_union_1.union_rows(dataset_to_union_2)
        >>> dataset_result = all_datasets[0].union_rows(*all_datasets[1:])
        >>> dataset_result = hl.MatrixTable.union_rows(*all_datasets)

        Notes
        -----

        In order to combine two datasets, three requirements must be met:

         - The column keys must be identical, both in type, value, and ordering.
         - The row key schemas and row schemas must match.
         - The entry schemas must match.

        The column fields in the resulting dataset are the column fields from
        the first dataset; the column schemas do not need to match.

        This method does not deduplicate; if a row exists identically in two
        datasets, then it will be duplicated in the result.

        Warning
        -------
        This method can trigger a shuffle, if partitions from two datasets
        overlap.

        Parameters
        ----------
        datasets : varargs of :class:`.MatrixTable`
            Datasets to combine.

        Returns
        -------
        :class:`.MatrixTable`
            Dataset with rows from each member of `datasets`.
        """
        if len(datasets) == 0:
            raise ValueError('Expected at least one argument')
        elif len(datasets) == 1:
            return datasets[0]
        else:
            error_msg = "'MatrixTable.union_rows' expects {} for all datasets to be the same. Found:    \ndataset {}: {}    \ndataset {}: {}"
            first = datasets[0]
            for i, next in enumerate(datasets[1:]):
                if first.row_key.keys() != next.row_key.keys():
                    raise ValueError(error_msg.format(
                        "row keys", 0, first.row_key.keys(), i+1, next.row_key.keys()
                    ))
                if first.row.dtype != next.row.dtype:
                    raise ValueError(error_msg.format(
                        "row types", 0, first.row.dtype, i+1, next.row.dtype
                    ))
                if first.entry.dtype != next.entry.dtype:
                    raise ValueError(error_msg.format(
                        "entry field types", 0, first.entry.dtype, i+1, next.entry.dtype
                    ))
                if first.col_key.dtype != next.col_key.dtype:
                    raise ValueError(error_msg.format(
                        "col key types", 0, first.col_key.dtype, i+1, next.col_key.dtype
                    ))
            return MatrixTable(MatrixUnionRows(*[d._mir for d in datasets]))

    @typecheck_method(other=matrix_table_type)
    def union_cols(self, other: 'MatrixTable') -> 'MatrixTable':
        """Take the union of dataset columns.

        Examples
        --------

        Union the columns of two datasets:

        >>> dataset_result = dataset_to_union_1.union_cols(dataset_to_union_2)

        Notes
        -----

        In order to combine two datasets, three requirements must be met:

         - The row keys must match.
         - The column key schemas and column schemas must match.
         - The entry schemas must match.

        The row fields in the resulting dataset are the row fields from the
        first dataset; the row schemas do not need to match.

        This method performs an inner join on rows and concatenates entries
        from the two datasets for each row.

        This method does not deduplicate; if a column key exists identically in
        two datasets, then it will be duplicated in the result.

        Parameters
        ----------
        other : :class:`.MatrixTable`
            Dataset to concatenate.

        Returns
        -------
        :class:`.MatrixTable`
            Dataset with columns from both datasets.
        """
        return MatrixTable._from_java(self._jmt.unionCols(other._jmt))

    @typecheck_method(n=int)
    def head(self, n: int) -> 'MatrixTable':
        """Subset matrix to first `n` rows.

        Examples
        --------
        Subset to the first three rows of the matrix:

        >>> dataset_result = dataset.head(3)
        >>> dataset_result.count_rows()
        3

        Notes
        -----

        The number of partitions in the new matrix is equal to the number of
        partitions containing the first `n` rows.

        Parameters
        ----------
        n : :obj:`int`
            Number of rows to include.

        Returns
        -------
        :class:`.MatrixTable`
            Matrix including the first `n` rows.
        """

        return MatrixTable._from_java(self._jmt.head(n))

    @typecheck_method(parts=sequenceof(int), keep=bool)
    def _filter_partitions(self, parts, keep=True):
        return MatrixTable._from_java(self._jmt.filterPartitions(parts, keep))

    @classmethod
    @typecheck_method(table=Table)
    def from_rows_table(cls, table: Table) -> 'MatrixTable':
        """Construct matrix table with no columns from a table.

        .. include:: _templates/experimental.rst

        Examples
        --------
        Import a text table and construct a rows-only matrix table:

        >>> table = hl.import_table('data/variant-lof.tsv')
        >>> table = table.transmute(**hl.parse_variant(table['v'])).key_by('locus', 'alleles')
        >>> sites_vds = hl.MatrixTable.from_rows_table(table)

        Notes
        -----
        All fields in the table become row-indexed fields in the
        result.

        Parameters
        ----------
        table : :class:`.Table`
            The table to be converted.

        Returns
        -------
        :class:`.MatrixTable`
        """
        hail.methods.misc.require_key(table, 'from_rows_table')
        jmt = scala_object(Env.hail().variant, 'MatrixTable').fromRowsTable(table._jt)
        return MatrixTable._from_java(jmt)

    @typecheck_method(p=numeric,
                      seed=nullable(int))
    def sample_rows(self, p: float, seed=None) -> 'MatrixTable':
        """Downsample the matrix table by keeping each row with probability ``p``.

        Examples
        --------

        Downsample the dataset to approximately 1% of its rows.

        >>> small_dataset = dataset.sample_rows(0.01)

        Parameters
        ----------
        p : :obj:`float`
            Probability of keeping each row.
        seed : :obj:`int`
            Random seed.

        Returns
        -------
        :class:`.MatrixTable`
            Matrix table with approximately ``p * n_rows`` rows.
        """

        if not (0 <= p <= 1):
            raise ValueError("Requires 'p' in [0,1]. Found p={}".format(p))

        return self.filter_rows(hl.rand_bool(p, seed))

    @typecheck_method(fields=dictof(str, str))
    def rename(self, fields: Dict[str, str]) -> 'MatrixTable':
        """Rename fields of a matrix table.

        Examples
        --------

        Rename column key `s` to `SampleID`, still keying by `SampleID`.

        >>> dataset_result = dataset.rename({'s': 'SampleID'})

        You can rename a field to a field name that already exists, as long as
        that field also gets renamed (no name collisions). Here, we rename the
        column key `s` to `info`, and the row field `info` to `vcf_info`:

        >>> dataset_result = dataset.rename({'s': 'info', 'info': 'vcf_info'})

        Parameters
        ----------
        fields : :obj:`dict` from :obj:`str` to :obj:`str`
            Mapping from old field names to new field names.

        Returns
        -------
        :class:`.MatrixTable`
            Matrix table with renamed fields.
        """

        seen = {}

        row_map = {}
        col_map = {}
        entry_map = {}
        global_map = {}

        for k, v in fields.items():
            if v in seen:
                raise ValueError(
                    "Cannot rename two fields to the same name: attempted to rename {} and {} both to {}".format(
                        repr(seen[v]), repr(k), repr(v)))
            if v in self._fields and v not in fields:
                raise ValueError("Cannot rename {} to {}: field already exists.".format(repr(k), repr(v)))
            seen[v] = k
            if self[k]._indices == self._row_indices:
                row_map[k] = v
            elif self[k]._indices == self._col_indices:
                col_map[k] = v
            elif self[k]._indices == self._entry_indices:
                entry_map[k] = v
            elif self[k]._indices == self._global_indices:
                global_map[k] = v

        return MatrixTable._from_java(self._jmt.renameFields(row_map, col_map, entry_map, global_map))

    def distinct_by_row(self):
        """Remove rows with a duplicate row key.

        Returns
        -------
        :class:`.MatrixTable`
        """
        return MatrixTable._from_java(self._jmt.distinctByRow())

    def distinct_by_col(self):
        """Remove columns with a duplicate row key.

        Returns
        -------
        :class:`.MatrixTable`
        """
        return MatrixTable._from_java(self._jmt.distinctByCol())

    @typecheck_method(separator=str)
    def make_table(self, separator='.') -> Table:
        """Make a table from a matrix table with one field per sample.

        Examples
        --------

        Consider a matrix table with the following schema:

        .. code-block:: text

          Global fields:
              'batch': str 
          Column fields:
              's': str 
          Row fields:
              'locus': locus<GRCh37> 
              'alleles': array<str> 
          Entry fields:
              'GT': call 
              'GQ': int32 
          Column key:
              's': str 
          Row key:
              'locus': locus<GRCh37> 
              'alleles': array<str>

        and three sample IDs: `A`, `B` and `C`.  Then the result of
        :func:`.make_table`:

        >>> ht = mt.make_table() # doctest: +SKIP

        has the original row fields along with 6 additional fields,
        one for each sample and entry field:

        .. code-block:: text

          Global fields:
              'batch': str 
          Row fields:
              'locus': locus<GRCh37> 
              'alleles': array<str> 
              'A.GT': call 
              'A.GQ': int32 
              'B.GT': call 
              'B.GQ': int32 
              'C.GT': call 
              'C.GQ': int32 
          Key:
              'locus': locus<GRCh37> 
              'alleles': array<str> 

        Notes
        -----

        The table has one row for each row of the input matrix.  The
        per sample and entry fields are formed by concatenating the
        sample ID with the entry field name using `separator`.  If the
        entry field name is empty, the separator is omitted.

        The table inherits the globals from the matrix table.

        Parameters
        ----------
        separator : :obj:`str`
            Separator between sample IDs and entry field names.

        Returns
        -------
        :class:`.Table`

        """
        return Table._from_java(self._jmt.makeTable(separator))

matrix_table_type.set(MatrixTable)
