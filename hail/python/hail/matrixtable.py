import itertools
from typing import Iterable, Optional, Dict, Tuple, Any, List
from collections import Counter
import hail as hl
from hail.expr.expressions import Expression, StructExpression, \
    expr_struct, expr_any, expr_bool, analyze, Indices, \
    construct_reference, construct_expr, extract_refs_by_indices, \
    ExpressionException, TupleExpression, unify_all
from hail.expr.types import types_match, tarray, tset
from hail.expr.matrix_type import tmatrix
import hail.ir as ir
from hail.table import Table, ExprContainer, TableIndexKeyError
from hail.typecheck import typecheck, typecheck_method, dictof, anytype, \
    anyfunc, nullable, sequenceof, oneof, numeric, lazy, enumeration
from hail.utils import storage_level, default_handler, deduplicate
from hail.utils.java import warning, Env, info
from hail.utils.misc import wrap_to_tuple, \
    get_key_by_exprs, \
    get_select_exprs, check_annotate_exprs, process_joins
import warnings


class GroupedMatrixTable(ExprContainer):
    """Matrix table grouped by row or column that can be aggregated into a new matrix table."""

    def __init__(self,
                 parent: 'MatrixTable',
                 row_keys=None,
                 computed_row_key=None,
                 col_keys=None,
                 computed_col_key=None,
                 entry_fields=None,
                 row_fields=None,
                 col_fields=None,
                 partitions=None):
        super(GroupedMatrixTable, self).__init__()
        self._parent = parent
        self._copy_fields_from(parent)
        self._row_keys = row_keys
        self._computed_row_key = computed_row_key
        self._col_keys = col_keys
        self._computed_col_key = computed_col_key
        self._entry_fields = entry_fields
        self._row_fields = row_fields
        self._col_fields = col_fields
        self._partitions = partitions

    def _copy(self, *,
              row_keys=None,
              computed_row_key=None,
              col_keys=None,
              computed_col_key=None,
              entry_fields=None,
              row_fields=None,
              col_fields=None,
              partitions=None):
        return GroupedMatrixTable(
            parent=self._parent,
            row_keys=row_keys if row_keys is not None else self._row_keys,
            computed_row_key=computed_row_key if computed_row_key is not None else self._computed_row_key,
            col_keys=col_keys if col_keys is not None else self._col_keys,
            computed_col_key=computed_col_key if computed_col_key is not None else self._computed_col_key,
            entry_fields=entry_fields if entry_fields is not None else self._entry_fields,
            row_fields=row_fields if row_fields is not None else self._row_fields,
            col_fields=col_fields if col_fields is not None else self._col_fields,
            partitions=partitions if partitions is not None else self._partitions
        )

    def _fixed_indices(self):
        if self._row_keys is None and self._col_keys is None:
            return self._parent._entry_indices
        if self._row_keys is not None and self._col_keys is None:
            return self._parent._col_indices
        if self._row_keys is None and self._col_keys is not None:
            return self._parent._row_indices
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

        s = (f'----------------------------------------\n'
             f'GroupedMatrixTable grouped by {rowstr}{colstr}\n'
             f'----------------------------------------\n'
             f'Parent MatrixTable:\n')

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
        ...                          .aggregate(n_non_ref = hl.agg.count_where(dataset.GT.is_non_ref())))

        Notes
        -----
        All complex expressions must be passed as named expressions.

        Parameters
        ----------
        exprs : args of :class:`str` or :class:`.Expression`
            Row fields to group by.
        named_exprs : keyword args of :class:`.Expression`
            Row-indexed expressions to group by.

        Returns
        -------
        :class:`.GroupedMatrixTable`
            Grouped matrix. Can be used to call :meth:`.GroupedMatrixTable.aggregate`.
        """
        if self._row_keys is not None:
            raise NotImplementedError("GroupedMatrixTable is already grouped by rows.")
        if self._col_keys is not None:
            raise NotImplementedError("GroupedMatrixTable is already grouped by cols; cannot also group by rows.")

        caller = 'group_rows_by'
        row_key, computed_key = get_key_by_exprs(caller,
                                                 exprs,
                                                 named_exprs,
                                                 self._parent._row_indices,
                                                 override_protected_indices={self._parent._global_indices,
                                                                             self._parent._col_indices})

        self._check_bindings(caller, computed_key, self._parent._row_indices)
        return self._copy(row_keys=row_key, computed_row_key=computed_key)

    @typecheck_method(exprs=oneof(str, Expression),
                      named_exprs=expr_any)
    def group_cols_by(self, *exprs, **named_exprs) -> 'GroupedMatrixTable':
        """Group columns.

        Examples
        --------
        Aggregate to a matrix with cohort as column keys, computing the call rate
        as an entry field:

        >>> dataset_result = (dataset.group_cols_by(dataset.cohort)
        ...                          .aggregate(call_rate = hl.agg.fraction(hl.is_defined(dataset.GT))))

        Notes
        -----
        All complex expressions must be passed as named expressions.

        Parameters
        ----------
        exprs : args of :class:`str` or :class:`.Expression`
            Column fields to group by.
        named_exprs : keyword args of :class:`.Expression`
            Column-indexed expressions to group by.

        Returns
        -------
        :class:`.GroupedMatrixTable`
            Grouped matrix, can be used to call :meth:`.GroupedMatrixTable.aggregate`.
        """
        if self._row_keys is not None:
            raise NotImplementedError("GroupedMatrixTable is already grouped by rows; cannot also group by cols.")
        if self._col_keys is not None:
            raise NotImplementedError("GroupedMatrixTable is already grouped by cols.")

        caller = 'group_cols_by'
        col_key, computed_key = get_key_by_exprs(caller,
                                                 exprs,
                                                 named_exprs,
                                                 self._parent._col_indices,
                                                 override_protected_indices={self._parent._global_indices,
                                                                             self._parent._row_indices})

        self._check_bindings(caller, computed_key, self._parent._col_indices)
        return self._copy(col_keys=col_key, computed_col_key=computed_key)

    def _check_bindings(self, caller, new_bindings, indices):
        empty = []

        def iter_option(o):
            return o if o is not None else empty

        if indices == self._parent._row_indices:
            fixed_fields = [*self._parent.globals, *self._parent.col]
        else:
            assert indices == self._parent._col_indices
            fixed_fields = [*self._parent.globals, *self._parent.row]

        bound_fields = set(itertools.chain(
            iter_option(self._row_keys),
            iter_option(self._col_keys),
            iter_option(self._col_fields),
            iter_option(self._row_fields),
            iter_option(self._entry_fields),
            fixed_fields))

        for k in new_bindings:
            if k in bound_fields:
                raise ExpressionException(f"{caller!r} cannot assign duplicate field {k!r}")

    def partition_hint(self, n: int) -> 'GroupedMatrixTable':
        """Set the target number of partitions for aggregation.

        Examples
        --------

        Use `partition_hint` in a :meth:`.MatrixTable.group_rows_by` /
        :meth:`.GroupedMatrixTable.aggregate` pipeline:

        >>> dataset_result = (dataset.group_rows_by(dataset.gene)
        ...                          .partition_hint(5)
        ...                          .aggregate(n_non_ref = hl.agg.count_where(dataset.GT.is_non_ref())))

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
        ...                          .aggregate_cols(mean_height = hl.agg.mean(dataset.pheno.height))
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
        if self._row_keys is not None:
            raise NotImplementedError("GroupedMatrixTable is already grouped by rows. Cannot aggregate over cols.")
        assert self._col_keys is not None

        base = self._col_fields if self._col_fields is not None else hl.struct()
        for k, e in named_exprs.items():
            analyze('GroupedMatrixTable.aggregate_cols', e, self._parent._global_indices, {self._parent._col_axis})

        self._check_bindings('aggregate_cols', named_exprs, self._parent._col_indices)
        return self._copy(col_fields=base.annotate(**named_exprs))

    @typecheck_method(named_exprs=expr_any)
    def aggregate_rows(self, **named_exprs) -> 'GroupedMatrixTable':
        """Aggregate rows by group.

        Examples
        --------
        Aggregate to a matrix with genes as row keys, collecting the functional
        consequences per gene as a set as a new row field:

        >>> dataset_result = (dataset.group_rows_by(dataset.gene)
        ...                          .aggregate_rows(consequences = hl.agg.collect_as_set(dataset.consequence))
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
        if self._col_keys is not None:
            raise NotImplementedError("GroupedMatrixTable is already grouped by cols. Cannot aggregate over rows.")
        assert self._row_keys is not None

        base = self._row_fields if self._row_fields is not None else hl.struct()
        for k, e in named_exprs.items():
            analyze('GroupedMatrixTable.aggregate_rows', e, self._parent._global_indices, {self._parent._row_axis})

        self._check_bindings('aggregate_rows', named_exprs, self._parent._row_indices)
        return self._copy(row_fields=base.annotate(**named_exprs))

    @typecheck_method(named_exprs=expr_any)
    def aggregate_entries(self, **named_exprs) -> 'GroupedMatrixTable':
        """Aggregate entries by group.

        Examples
        --------
        Aggregate to a matrix with genes as row keys, computing the number of
        non-reference calls as an entry field:

        >>> dataset_result = (dataset.group_rows_by(dataset.gene)
        ...                          .aggregate_entries(n_non_ref = hl.agg.count_where(dataset.GT.is_non_ref()))
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

        base = self._entry_fields if self._entry_fields is not None else hl.struct()
        for k, e in named_exprs.items():
            analyze('GroupedMatrixTable.aggregate_entries', e, self._fixed_indices(), {self._parent._row_axis, self._parent._col_axis})

        self._check_bindings('aggregate_entries', named_exprs,
                             self._parent._col_indices if self._col_keys is not None else self._parent._row_indices)
        return self._copy(entry_fields=base.annotate(**named_exprs))

    def result(self) -> 'MatrixTable':
        """Return the result of aggregating by group.

        Examples
        --------
        Aggregate to a matrix with genes as row keys, collecting the functional
        consequences per gene as a row field and computing the number of
        non-reference calls as an entry field:

        >>> dataset_result = (dataset.group_rows_by(dataset.gene)
        ...                          .aggregate_rows(consequences = hl.agg.collect_as_set(dataset.consequence))
        ...                          .aggregate_entries(n_non_ref = hl.agg.count_where(dataset.GT.is_non_ref()))
        ...                          .result())

        Aggregate to a matrix with cohort as column keys, computing the mean height
        per cohort as a column field and computing the number of non-reference calls
        as an entry field:

        >>> dataset_result = (dataset.group_cols_by(dataset.cohort)
        ...                          .aggregate_cols(mean_height = hl.agg.stats(dataset.pheno.height).mean)
        ...                          .aggregate_entries(n_non_ref = hl.agg.count_where(dataset.GT.is_non_ref()))
        ...                          .result())

        See Also
        --------
        :meth:`.aggregate`

        Returns
        -------
        :class:`.MatrixTable`
            Aggregated matrix table.
        """
        assert self._row_keys is not None or self._col_keys is not None

        defined_exprs = []
        for e in [self._row_fields, self._col_fields, self._entry_fields]:
            if e is not None:
                defined_exprs.append(e)
        for e in [self._computed_row_key, self._computed_col_key]:
            if e is not None:
                defined_exprs.extend(e.values())

        def promote_none(e):
            return hl.struct() if e is None else e
        entry_exprs = promote_none(self._entry_fields)
        if len(entry_exprs) == 0:
            warning("'GroupedMatrixTable.result': No entry fields were defined.")

        base, cleanup = self._parent._process_joins(*defined_exprs)

        if self._col_keys is not None:
            cck = self._computed_col_key or {}
            computed_key_uids = {k: Env.get_uid() for k in cck}
            modified_keys = [computed_key_uids.get(k, k) for k in self._col_keys]
            mt = MatrixTable(ir.MatrixAggregateColsByKey(
                ir.MatrixMapCols(
                    base._mir,
                    self._parent.col.annotate(**{computed_key_uids[k]: v for k, v in cck.items()})._ir,
                    modified_keys),
                entry_exprs._ir,
                promote_none(self._col_fields)._ir))
            if cck:
                mt = mt.rename({v: k for k, v in computed_key_uids.items()})
        else:
            cck = self._computed_row_key or {}
            computed_key_uids = {k: Env.get_uid() for k in cck}
            modified_keys = [computed_key_uids.get(k, k) for k in self._row_keys]
            mt = MatrixTable(ir.MatrixAggregateRowsByKey(
                ir.MatrixKeyRowsBy(
                    ir.MatrixMapRows(
                        ir.MatrixKeyRowsBy(base._mir, []),
                        self._parent._rvrow.annotate(**{computed_key_uids[k]: v for k, v in cck.items()})._ir),
                    modified_keys),
                entry_exprs._ir,
                promote_none(self._row_fields)._ir))
            if cck:
                mt = mt.rename({v: k for k, v in computed_key_uids.items()})

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
        ...                          .aggregate(n_non_ref = hl.agg.count_where(dataset.GT.is_non_ref())))

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

    >>> dataset = dataset.annotate_globals(pli = {'SCN1A': 0.999, 'SONIC': 0.014},
    ...                                    populations = ['AFR', 'EAS', 'EUR', 'SAS', 'AMR', 'HIS'])

    >>> dataset = dataset.annotate_cols(pop = dataset.populations[hl.int(hl.rand_unif(0, 6))],
    ...                                 sample_gq = hl.agg.mean(dataset.GQ),
    ...                                 sample_dp = hl.agg.mean(dataset.DP))

    >>> dataset = dataset.annotate_rows(variant_gq = hl.agg.mean(dataset.GQ),
    ...                                 variant_dp = hl.agg.mean(dataset.GQ),
    ...                                 sas_hets = hl.agg.count_where(dataset.GT.is_het()))

    >>> dataset = dataset.annotate_entries(gq_by_dp = dataset.GQ / dataset.DP)

    Filter:

    >>> dataset = dataset.filter_cols(dataset.pop != 'EUR')

    >>> datasetm = dataset.filter_rows((dataset.variant_gq > 10) & (dataset.variant_dp > 5))

    >>> dataset = dataset.filter_entries(dataset.gq_by_dp > 1)

    Query:

    >>> col_stats = dataset.aggregate_cols(hl.struct(pop_counts=hl.agg.counter(dataset.pop),
    ...                                              high_quality=hl.agg.fraction((dataset.sample_gq > 10) & (dataset.sample_dp > 5))))
    >>> print(col_stats.pop_counts)
    >>> print(col_stats.high_quality)

    >>> het_dist = dataset.aggregate_rows(hl.agg.stats(dataset.sas_hets))
    >>> print(het_dist)

    >>> entry_stats = dataset.aggregate_entries(hl.struct(call_rate=hl.agg.fraction(hl.is_defined(dataset.GT)),
    ...                                                   global_gq_mean=hl.agg.mean(dataset.GQ)))
    >>> print(entry_stats.call_rate)
    >>> print(entry_stats.global_gq_mean)
    """

    @staticmethod
    @typecheck(
        globals=nullable(dictof(str, anytype)),
        rows=nullable(dictof(str, sequenceof(anytype))),
        cols=nullable(dictof(str, sequenceof(anytype))),
        entries=nullable(dictof(str, sequenceof(sequenceof(anytype)))),
    )
    def from_parts(
        globals: Optional[Dict[str, Any]] = None,
        rows: Optional[Dict[str, Iterable[Any]]] = None,
        cols: Optional[Dict[str, Iterable[Any]]] = None,
        entries: Optional[Dict[str, Iterable[Iterable[Any]]]] = None
    ) -> 'MatrixTable':
        """Create a `MatrixTable` from its component parts.

        Example
        -------
        >>> mt = hl.MatrixTable.from_parts(
        ...     globals={'hello':'world'},
        ...     rows={'foo':[1, 2]},
        ...     cols={'bar':[3, 4]},
        ...     entries={'baz':[[1, 2],[3, 4]]}
        ... )
        >>> mt.describe()
        ----------------------------------------
        Global fields:
            'hello': str
        ----------------------------------------
        Column fields:
            'col_idx': int32
            'bar': int32
        ----------------------------------------
        Row fields:
            'row_idx': int32
            'foo': int32
        ----------------------------------------
        Entry fields:
            'baz': int32
        ----------------------------------------
        Column key: ['col_idx']
        Row key: ['row_idx']
        ----------------------------------------
        >>> mt.row.show()
        +---------+-------+
        | row_idx |   foo |
        +---------+-------+
        |   int32 | int32 |
        +---------+-------+
        |       0 |     1 |
        |       1 |     2 |
        +---------+-------+
        >>> mt.col.show()
        +---------+-------+
        | col_idx |   bar |
        +---------+-------+
        |   int32 | int32 |
        +---------+-------+
        |       0 |     3 |
        |       1 |     4 |
        +---------+-------+
        >>> mt.entry.show()
        +---------+-------+-------+
        | row_idx | 0.baz | 1.baz |
        +---------+-------+-------+
        |   int32 | int32 | int32 |
        +---------+-------+-------+
        |       0 |     1 |     2 |
        |       1 |     3 |     4 |
        +---------+-------+-------+

        Notes
        -----
        - Matrix dimensions are inferred from input data.
        - You must provide row and column dimensions by specifying rows or
          entries (inclusive) and cols or entries (inclusive).
        - The respective dimensions of rows, cols and entries must match should
          you provide rows and entries or cols and entries (inclusive).

        Parameters
        ----------
        globals : :class:`dict` from :class:`str` to :obj:`any`
            Global fields by name.

        rows: :class:`dict` from :class:`str` to :class:`list` of :obj:`any`
            Row fields by name.

        cols: :class:`dict` from :class:`str` to :class:`list` of :obj:`any`
            Column fields by name.

        entries: :class:`dict` from :class:`str` to :class:`list` of :class:`list` of :obj:`any`
            Matrix entries by name in the form `entry[row_idx][col_idx]`.

        Returns
        -------
        :class:`.MatrixTable`
            A MatrixTable assembled from inputs whose rows are keyed by `row_idx`
            and columns are keyed by `col_idx`.
        """
        # General idea: build a `Table` representation matching that returned by
        # `MatrixTable.localize_entries` and then call `_unlocalize_entries`. In
        # this form, the column table is bundled with the globals and the entries
        # for each row is stored on the row.
        def raise_when_mismatched_property_dimensions(kvs: Dict[str, Iterable[Any]]):
            def value_len(entry):
                return len(entry[1])

            kvs = sorted(kvs.items(), key=value_len)
            dims = itertools.groupby(kvs, value_len)
            dims = {size: [k for k, _ in group] for size, group in dims}
            if len(dims) > 1:
                raise ValueError(f"property matrix dimensions do not match: {dims}.")

        def transpose(kvs: Dict[str, Iterable[Any]]) -> List[Dict[str, Any]]:
            raise_when_mismatched_property_dimensions(kvs)
            return [dict(zip(kvs, vs)) for vs in zip(*kvs.values())]

        def anyval(kvs):
            return next(iter(kvs.values()))

        # In the case rows or cols aren't specified, we need to infer the
        # matrix dimensions from *an* entry. Which one isn't important as we
        # enforce congruence among input dimensions.
        assert not ((rows is None or cols is None) and (entries is None))
        cols = transpose(cols) if cols else [{} for _ in anyval(entries)[0]]
        for i, _ in enumerate(cols):
            cols[i] = hl.struct(col_idx=i, **cols[i])

        if globals is None:
            globals = {}

        cols_field_name = Env.get_uid()
        globals[cols_field_name] = cols

        rows = transpose(rows) if rows else [{} for _ in anyval(entries)]
        entries = [transpose(e) for e in transpose(entries)
                   ] if entries else [[{} for _ in cols] for _ in rows]

        if len(rows) != len(entries) or len(cols) != len(entries[0]):
            raise ValueError((
                "mismatched matrix dimensions: "
                "number of rows and cols does not match entry dimensions."
            ))

        entries_field_name = Env.get_uid()
        for i, (row_props, entry_props) in enumerate(zip(rows, entries)):
            row_entries = [hl.struct(**kvs) for kvs in entry_props]
            rows[i] = hl.Struct(row_idx=i, **row_props, **{entries_field_name: row_entries})

        ht = Table.parallelize(rows, key='row_idx', globals=hl.struct(**globals))
        return ht._unlocalize_entries(entries_field_name, cols_field_name, col_key=['col_idx'])

    def __init__(self, mir):
        super(MatrixTable, self).__init__()

        self._mir = mir

        self._globals = None
        self._col_values = None

        self._row_axis = 'row'
        self._col_axis = 'column'

        self._global_indices = Indices(self, set())
        self._row_indices = Indices(self, {self._row_axis})
        self._col_indices = Indices(self, {self._col_axis})
        self._entry_indices = Indices(self, {self._row_axis, self._col_axis})

        self._type = self._mir.typ

        self._global_type = self._type.global_type
        self._col_type = self._type.col_type
        self._row_type = self._type.row_type
        self._entry_type = self._type.entry_type

        self._globals = construct_reference('global', self._global_type,
                                            indices=self._global_indices)
        self._rvrow = construct_reference('va',
                                          self._type.row_type,
                                          indices=self._row_indices)
        self._row = hl.struct(**{k: self._rvrow[k] for k in self._row_type.keys()})
        self._col = construct_reference('sa', self._col_type,
                                        indices=self._col_indices)
        self._entry = construct_reference('g', self._entry_type,
                                          indices=self._entry_indices)

        self._indices_from_ref = {'global': self._global_indices,
                                  'va': self._row_indices,
                                  'sa': self._col_indices,
                                  'g': self._entry_indices}

        self._row_key = hl.struct(
            **{k: self._row[k] for k in self._type.row_key})
        self._partition_key = self._row_key
        self._col_key = hl.struct(
            **{k: self._col[k] for k in self._type.col_key})

        self._num_samples = None

        for k, v in itertools.chain(self._globals.items(),
                                    self._row.items(),
                                    self._col.items(),
                                    self._entry.items()):
            self._set_field(k, v)

    @property
    def _schema(self) -> tmatrix:
        return tmatrix(
            self._global_type,
            self._col_type, list(self._col_key),
            self._row_type, list(self._row_key),
            self._entry_type)

    def __getitem__(self, item):
        invalid_usage = TypeError("MatrixTable.__getitem__: invalid index argument(s)\n"
                                  "  Usage 1: field selection: mt['field']\n"
                                  "  Usage 2: Entry joining: mt[mt2.row_key, mt2.col_key]\n\n"
                                  "  To join row or column fields, use one of the following:\n"
                                  "    rows:\n"
                                  "       mt.index_rows(mt2.row_key)\n"
                                  "       mt.rows().index(mt2.row_key)\n"
                                  "       mt.rows()[mt2.row_key]\n"
                                  "    cols:\n"
                                  "       mt.index_cols(mt2.col_key)\n"
                                  "       mt.cols().index(mt2.col_key)\n"
                                  "       mt.cols()[mt2.col_key]")

        if isinstance(item, str):
            return self._get_field(item)
        if isinstance(item, tuple) and len(item) == 2:
            # this is the join path
            exprs = item
            row_key = wrap_to_tuple(exprs[0])
            col_key = wrap_to_tuple(exprs[1])

            try:
                return self.index_entries(row_key, col_key)
            except TypeError as e:
                raise invalid_usage from e
        raise invalid_usage

    @property
    def _col_key_types(self):
        return [v.dtype for _, v in self.col_key.items()]

    @property
    def _row_key_types(self):
        return [v.dtype for _, v in self.row_key.items()]

    @property
    def col_key(self) -> 'StructExpression':
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
    def row_key(self) -> 'StructExpression':
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
    def globals(self) -> 'StructExpression':
        """Returns a struct expression including all global fields.

        Returns
        -------
        :class:`.StructExpression`
        """
        return self._globals

    @property
    def row(self) -> 'StructExpression':
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
    def row_value(self) -> 'StructExpression':
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
    def col(self) -> 'StructExpression':
        """Returns a struct expression of all column-indexed fields, including keys.

        Examples
        --------
        Get all column field names:

        >>> list(dataset.col)  # doctest: +SKIP_OUTPUT_CHECK
        ['s', 'sample_qc', 'is_case', 'pheno', 'cov', 'cov1', 'cov2', 'cohorts', 'pop']

        Returns
        -------
        :class:`.StructExpression`
            Struct of all column fields.
        """
        return self._col

    @property
    def col_value(self) -> 'StructExpression':
        """Returns a struct expression including all non-key column-indexed fields.

        Examples
        --------
        Get all non-key column field names:

        >>> list(dataset.col_value)  # doctest: +SKIP_OUTPUT_CHECK
        ['sample_qc', 'is_case', 'pheno', 'cov', 'cov1', 'cov2', 'cohorts', 'pop']

        Returns
        -------
        :class:`.StructExpression`
            Struct of all column fields, minus keys.
        """
        return self._col.drop(*self.col_key)

    @property
    def entry(self) -> 'StructExpression':
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
        keys : varargs of :class:`str` or :class:`.Expression`.
            Column fields to key by.
        named_keys : keyword args of :class:`.Expression`.
            Column fields to key by.
        Returns
        -------
        :class:`.MatrixTable`
        """
        key_fields, computed_keys = get_key_by_exprs("MatrixTable.key_cols_by", keys, named_keys, self._col_indices)

        if not computed_keys:
            return MatrixTable(ir.MatrixMapCols(self._mir, self._col._ir, key_fields))
        else:
            new_col = self.col.annotate(**computed_keys)
            base, cleanup = self._process_joins(new_col)

            return cleanup(MatrixTable(
                ir.MatrixMapCols(
                    base._mir,
                    new_col._ir,
                    key_fields
                )))

    @typecheck_method(new_key=str)
    def _key_rows_by_assert_sorted(self, *new_key):
        rk_names = list(self.row_key)
        i = 0
        while (i < min(len(new_key), len(rk_names))):
            if new_key[i] != rk_names[i]:
                break
            i += 1

        if i < 1:
            raise ValueError(
                f'cannot implement an unsafe sort with no shared key:\n  new key: {new_key}\n  old key: {rk_names}')

        return MatrixTable(ir.MatrixKeyRowsBy(self._mir, list(new_key), is_sorted=True))

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
        keys : varargs of :class:`str` or :class:`.Expression`.
            Row fields to key by.
        named_keys : keyword args of :class:`.Expression`.
            Row fields to key by.
        Returns
        -------
        :class:`.MatrixTable`
        """
        key_fields, computed_keys = get_key_by_exprs("MatrixTable.key_rows_by", keys, named_keys, self._row_indices)

        if not computed_keys:
            return MatrixTable(ir.MatrixKeyRowsBy(self._mir, key_fields))
        else:
            new_row = self._rvrow.annotate(**computed_keys)
            base, cleanup = self._process_joins(new_row)

            return cleanup(MatrixTable(
                ir.MatrixKeyRowsBy(
                    ir.MatrixMapRows(
                        ir.MatrixKeyRowsBy(base._mir, []),
                        new_row._ir),
                    list(key_fields))))

    @typecheck_method(named_exprs=expr_any)
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

        caller = "MatrixTable.annotate_globals"
        check_annotate_exprs(caller, named_exprs, self._global_indices, set())
        return self._select_globals(caller, self.globals.annotate(**named_exprs))

    @typecheck_method(named_exprs=expr_any)
    def annotate_rows(self, **named_exprs) -> 'MatrixTable':
        """Create new row-indexed fields by name.

        Examples
        --------
        Compute call statistics for high quality samples per variant:

        >>> high_quality_calls = hl.agg.filter(dataset.sample_qc.gq_stats.mean > 20,
        ...                                    hl.agg.call_stats(dataset.GT, dataset.alleles))
        >>> dataset_result = dataset.annotate_rows(call_stats = high_quality_calls)

        Add functional annotations from a :class:`.Table`, `v_metadata`, and a
        :class:`.MatrixTable`, `dataset2_AF`, both keyed by locus and alleles.

        >>> dataset_result = dataset.annotate_rows(consequence = v_metadata[dataset.locus, dataset.alleles].consequence,
        ...                                        dataset2_AF = dataset2.index_rows(dataset.row_key).info.AF)

        Note
        ----
        This method supports aggregation over columns. For instance, the usage:

        >>> dataset_result = dataset.annotate_rows(mean_GQ = hl.agg.mean(dataset.GQ))

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
        check_annotate_exprs(caller, named_exprs, self._row_indices, {self._col_axis})
        return self._select_rows(caller, self._rvrow.annotate(**named_exprs))

    @typecheck_method(named_exprs=expr_any)
    def annotate_cols(self, **named_exprs) -> 'MatrixTable':
        """Create new column-indexed fields by name.

        Examples
        --------
        Compute statistics about the GQ distribution per sample:

        >>> dataset_result = dataset.annotate_cols(sample_gq_stats = hl.agg.stats(dataset.GQ))

        Add sample metadata from a :class:`.hail.Table`.

        >>> dataset_result = dataset.annotate_cols(population = s_metadata[dataset.s].pop)

        Note
        ----
        This method supports aggregation over rows. For instance, the usage:

        >>> dataset_result = dataset.annotate_cols(mean_GQ = hl.agg.mean(dataset.GQ))

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
        check_annotate_exprs(caller, named_exprs, self._col_indices, {self._row_axis})
        return self._select_cols(caller, self.col.annotate(**named_exprs))

    @typecheck_method(named_exprs=expr_any)
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
        check_annotate_exprs(caller, named_exprs, self._entry_indices, set())
        return self._select_entries(caller, s=self.entry.annotate(**named_exprs))

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
        exprs : variable-length args of :class:`str` or :class:`.Expression`
            Arguments that specify field names or nested field reference expressions.
        named_exprs : keyword args of :class:`.Expression`
            Field names and the expressions to compute them.

        Returns
        -------
        :class:`.MatrixTable`
            MatrixTable with specified global fields.
        """

        caller = 'MatrixTable.select_globals'
        new_global = get_select_exprs(caller,
                                      exprs,
                                      named_exprs,
                                      self._global_indices,
                                      self._globals)
        return self._select_globals(caller, new_global)

    def select_rows(self, *exprs, **named_exprs) -> 'MatrixTable':
        """Select existing row fields or create new fields by name, dropping all
        other non-key fields.

        Examples
        --------
        Select existing fields and compute a new one:

        >>> dataset_result = dataset.select_rows(
        ...    dataset.variant_qc.gq_stats.mean,
        ...    high_quality_cases = hl.agg.count_where((dataset.GQ > 20) &
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

        >>> dataset_result = dataset.select_rows(mean_GQ = hl.agg.mean(dataset.GQ))

        will compute the mean per row.

        Parameters
        ----------
        exprs : variable-length args of :class:`str` or :class:`.Expression`
            Arguments that specify field names or nested field reference expressions.
        named_exprs : keyword args of :class:`.Expression`
            Field names and the expressions to compute them.

        Returns
        -------
        :class:`.MatrixTable`
            MatrixTable with specified row fields.
        """
        caller = 'MatrixTable.select_rows'
        new_row = get_select_exprs(caller,
                                   exprs,
                                   named_exprs,
                                   self._row_indices,
                                   self._rvrow)
        return self._select_rows(caller, new_row)

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

        >>> dataset_result = dataset.select_cols(mean_GQ = hl.agg.mean(dataset.GQ))

        will compute the mean per column.

        Parameters
        ----------
        exprs : variable-length args of :class:`str` or :class:`.Expression`
            Arguments that specify field names or nested field reference expressions.
        named_exprs : keyword args of :class:`.Expression`
            Field names and the expressions to compute them.

        Returns
        -------
        :class:`.MatrixTable`
            MatrixTable with specified column fields.
        """
        caller = 'MatrixTable.select_cols'
        new_col = get_select_exprs(caller,
                                   exprs,
                                   named_exprs,
                                   self._col_indices,
                                   self._col)
        return self._select_cols(caller, new_col)

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
        exprs : variable-length args of :class:`str` or :class:`.Expression`
            Arguments that specify field names or nested field reference expressions.
        named_exprs : keyword args of :class:`.Expression`
            Field names and the expressions to compute them.

        Returns
        -------
        :class:`.MatrixTable`
            MatrixTable with specified entry fields.
        """
        caller = 'MatrixTable.select_entries'
        new_entry = get_select_exprs(caller,
                                     exprs,
                                     named_exprs,
                                     self._entry_indices,
                                     self._entry)
        return self._select_entries(caller, new_entry)

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
        exprs : varargs of :class:`str` or :class:`.Expression`
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
                    raise ExpressionException("Method 'drop' expects string field names or top-level field expressions"
                                              " (e.g. 'foo', matrix.foo, or matrix['foo'])")
            else:
                assert isinstance(e, str)
                if e not in self._fields:
                    raise IndexError("MatrixTable has no field '{}'".format(e))
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

    @typecheck_method(other=Table)
    def semi_join_rows(self, other: 'Table') -> 'MatrixTable':
        """Filters the matrix table to rows whose key appears in `other`.

        Parameters
        ----------
        other : :class:`.Table`
            Table with compatible key field(s).

        Returns
        -------
        :class:`.MatrixTable`

        Notes
        -----
        The row key type of the matrix table must match the key type of `other`.

        This method does not change the schema of the matrix table; it is
        filtering the matrix table to row keys present in another table.

        To discard rows whose key is present in `other`, use
        :meth:`.anti_join_rows`.

        Examples
        --------
        >>> ds_result = ds.semi_join_rows(rows_to_keep)

        It may be expensive to key the matrix table by the right-side key.
        In this case, it is possible to implement a semi-join using a non-key
        field as follows:

        >>> ds_result = ds.filter_rows(hl.is_defined(rows_to_keep.index(ds['locus'], ds['alleles'])))

        See Also
        --------
        :meth:`.anti_join_rows`, :meth:`.filter_rows`, :meth:`.semi_join_cols`
        """
        if len(other.key) == 0:
            raise ValueError('semi_join_rows: cannot join with a table with no key')
        if len(other.key) > len(self.row_key) or any(t[0].dtype != t[1].dtype for t in zip(self.row_key.values(), other.key.values())):
            raise ValueError('semi_join_rows: cannot join: table must have a key of the same type(s) and be the same length or shorter:'
                             f'\n  MatrixTable row key: {", ".join(str(x.dtype) for x in self.row_key.values())}'
                             f'\n            Table key: {", ".join(str(x.dtype) for x in other.key.values())}')
        return self.filter_rows(hl.is_defined(other.index(*(self.row_key[i] for i in range(len(other.key))))))

    @typecheck_method(other=Table)
    def anti_join_rows(self, other: 'Table') -> 'MatrixTable':
        """Filters the table to rows whose key does not appear in `other`.

        Parameters
        ----------
        other : :class:`.Table`
            Table with compatible key field(s).

        Returns
        -------
        :class:`.MatrixTable`

        Notes
        -----
        The row key type of the matrix table must match the key type of `other`.

        This method does not change the schema of the table; it is a method of
        filtering the matrix table to row keys not present in another table.

        To restrict to rows whose key is present in `other`, use
        :meth:`.semi_join_rows`.

        Examples
        --------
        >>> ds_result = ds.anti_join_rows(rows_to_remove)

        It may be expensive to key the matrix table by the right-side key.
        In this case, it is possible to implement an anti-join using a non-key
        field as follows:

        >>> ds_result = ds.filter_rows(hl.is_missing(rows_to_remove.index(ds['locus'], ds['alleles'])))

        See Also
        --------
        :meth:`.anti_join_rows`, :meth:`.filter_rows`, :meth:`.anti_join_cols`
        """
        if len(other.key) == 0:
            raise ValueError('anti_join_rows: cannot join with a table with no key')
        if len(other.key) > len(self.row_key) or any(t[0].dtype != t[1].dtype for t in zip(self.row_key.values(), other.key.values())):
            raise ValueError('anti_join_rows: cannot join: table must have a key of the same type(s) and be the same length or shorter:'
                             f'\n  MatrixTable row key: {", ".join(str(x.dtype) for x in self.row_key.values())}'
                             f'\n            Table key: {", ".join(str(x.dtype) for x in other.key.values())}')
        return self.filter_rows(hl.is_missing(other.index(*(self.row_key[i] for i in range(len(other.key))))))

    @typecheck_method(other=Table)
    def semi_join_cols(self, other: 'Table') -> 'MatrixTable':
        """Filters the matrix table to columns whose key appears in `other`.

        Parameters
        ----------
        other : :class:`.Table`
            Table with compatible key field(s).

        Returns
        -------
        :class:`.MatrixTable`

        Notes
        -----
        The column key type of the matrix table must match the key type of `other`.

        This method does not change the schema of the matrix table; it is a
        filtering the matrix table to column keys not present in another table.

        To discard collumns whose key is present in `other`, use
        :meth:`.anti_join_cols`.

        Examples
        --------
        >>> ds_result = ds.semi_join_cols(cols_to_keep)

        It may be inconvenient to key the matrix table by the right-side key.
        In this case, it is possible to implement a semi-join using a non-key
        field as follows:

        >>> ds_result = ds.filter_cols(hl.is_defined(cols_to_keep.index(ds['s'])))

        See Also
        --------
        :meth:`.anti_join_cols`, :meth:`.filter_cols`, :meth:`.semi_join_rows`
        """
        if len(other.key) == 0:
            raise ValueError('semi_join_cols: cannot join with a table with no key')
        if len(other.key) > len(self.col_key) or any(t[0].dtype != t[1].dtype for t in zip(self.col_key.values(), other.key.values())):
            raise ValueError('semi_join_cols: cannot join: table must have a key of the same type(s) and be the same length or shorter:'
                             f'\n  MatrixTable col key: {", ".join(str(x.dtype) for x in self.col_key.values())}'
                             f'\n            Table key: {", ".join(str(x.dtype) for x in other.key.values())}')

        return self.filter_cols(hl.is_defined(other.index(*(self.col_key[i] for i in range(len(other.key))))))

    @typecheck_method(other=Table)
    def anti_join_cols(self, other: 'Table') -> 'MatrixTable':
        """Filters the table to columns whose key does not appear in `other`.

        Parameters
        ----------
        other : :class:`.Table`
            Table with compatible key field(s).

        Returns
        -------
        :class:`.MatrixTable`

        Notes
        -----
        The column key type of the matrix table must match the key type of `other`.

        This method does not change the schema of the table; it is a method of
        filtering the matrix table to column keys not present in another table.

        To restrict to columns whose key is present in `other`, use
        :meth:`.semi_join_cols`.

        Examples
        --------
        >>> ds_result = ds.anti_join_cols(cols_to_remove)

        It may be inconvenient to key the matrix table by the right-side key.
        In this case, it is possible to implement an anti-join using a non-key
        field as follows:

        >>> ds_result = ds.filter_cols(hl.is_missing(cols_to_remove.index(ds['s'])))

        See Also
        --------
        :meth:`.semi_join_cols`, :meth:`.filter_cols`, :meth:`.anti_join_rows`
        """
        if len(other.key) == 0:
            raise ValueError('anti_join_cols: cannot join with a table with no key')
        if len(other.key) > len(self.col_key) or any(t[0].dtype != t[1].dtype for t in zip(self.col_key.values(), other.key.values())):
            raise ValueError('anti_join_cols: cannot join: table must have a key of the same type(s) and be the same length or shorter:'
                             f'\n  MatrixTable col key: {", ".join(str(x.dtype) for x in self.col_key.values())}'
                             f'\n            Table key: {", ".join(str(x.dtype) for x in other.key.values())}')

        return self.filter_cols(hl.is_missing(other.index(*(self.col_key[i] for i in range(len(other.key))))))

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
        The expression `expr` will be evaluated for every row of the table. If
        `keep` is ``True``, then rows where `expr` evaluates to ``True`` will be
        kept (the filter removes the rows where the predicate evaluates to
        ``False``). If `keep` is ``False``, then rows where `expr` evaluates to
        ``True`` will be removed (the filter keeps the rows where the predicate
        evaluates to ``False``).

        Warning
        -------
        When `expr` evaluates to missing, the row will be removed regardless of `keep`.

        Note
        ----
        This method supports aggregation over columns. For instance,

        >>> dataset_result = dataset.filter_rows(hl.agg.mean(dataset.GQ) > 20.0)

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
        mt = MatrixTable(ir.MatrixFilterRows(base._mir, ir.filter_predicate_with_keep(expr._ir, keep)))
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
        The expression `expr` will be evaluated for every column of the table.
        If `keep` is ``True``, then columns where `expr` evaluates to ``True``
        will be kept (the filter removes the columns where the predicate
        evaluates to ``False``). If `keep` is ``False``, then columns where
        `expr` evaluates to ``True`` will be removed (the filter keeps the
        columns where the predicate evaluates to ``False``).

        Warning
        -------
        When `expr` evaluates to missing, the column will be removed regardless of
        `keep`.

        Note
        ----
        This method supports aggregation over rows. For instance,

        >>> dataset_result = dataset.filter_cols(hl.agg.mean(dataset.GQ) > 20.0)

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
        mt = MatrixTable(ir.MatrixFilterCols(base._mir, ir.filter_predicate_with_keep(expr._ir, keep)))
        return cleanup(mt)

    @typecheck_method(expr=expr_bool, keep=bool)
    def filter_entries(self, expr, keep: bool = True) -> 'MatrixTable':
        """Filter entries of the matrix.

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

        Examples
        --------

        Keep entries where the sum of `AD` is greater than 10 and `GQ` is greater than 20:

        >>> dataset_result = dataset.filter_entries((hl.sum(dataset.AD) > 10) & (dataset.GQ > 20))

        Warning
        -------
        When `expr` evaluates to missing, the entry will be removed regardless of
        `keep`.

        Note
        ----
        This method does not support aggregation.

        Notes
        -----
        The expression `expr` will be evaluated for every entry of the table.
        If `keep` is ``True``, then entries where `expr` evaluates to ``True``
        will be kept (the filter removes the entries where the predicate
        evaluates to ``False``). If `keep` is ``False``, then entries where
        `expr` evaluates to ``True`` will be removed (the filter keeps the
        entries where the predicate evaluates to ``False``).

        Filtered entries are removed entirely from downstream operations. This
        means that the resulting matrix table has sparsity -- that is, that the
        number of entries is **smaller** than the product of :meth:`count_rows`
        and :meth:`count_cols`. To re-densify a filtered matrix table, use the
        :meth:`unfilter_entries` method to restore filtered entries, populated
        all fields with missing values. Below are some properties of an
        entry-filtered matrix table.

        1. Filtered entries are not included in the :meth:`entries` table.

        >>> mt_range = hl.utils.range_matrix_table(10, 10)
        >>> mt_range = mt_range.annotate_entries(x = mt_range.row_idx + mt_range.col_idx)
        >>> mt_range.count()
        (10, 10)

        >>> mt_range.entries().count()
        100

        >>> mt_filt = mt_range.filter_entries(mt_range.x % 2 == 0)
        >>> mt_filt.count()
        (10, 10)

        >>> mt_filt.count_rows() * mt_filt.count_cols()
        100

        >>> mt_filt.entries().count()
        50

        2. Filtered entries are not included in aggregation.

        >>> mt_filt.aggregate_entries(hl.agg.count())
        50

        >>> mt_filt = mt_filt.annotate_cols(col_n = hl.agg.count())
        >>> mt_filt.col_n.take(5)
        [5, 5, 5, 5, 5]

        >>> mt_filt = mt_filt.annotate_rows(row_n = hl.agg.count())
        >>> mt_filt.row_n.take(5)
        [5, 5, 5, 5, 5]

        3. Annotating a new entry field will not annotate filtered entries.

        >>> mt_filt = mt_filt.annotate_entries(y = 1)
        >>> mt_filt.aggregate_entries(hl.agg.sum(mt_filt.y))
        50

        4. If all the entries in a row or column of a matrix table are
        filtered, the row or column remains.

        >>> mt_filt.filter_entries(False).count()
        (10, 10)

        See Also
        --------
        :meth:`unfilter_entries`, :meth:`compute_entry_filter_stats`
        """
        base, cleanup = self._process_joins(expr)
        analyze('MatrixTable.filter_entries', expr, self._entry_indices)

        m = MatrixTable(ir.MatrixFilterEntries(base._mir, ir.filter_predicate_with_keep(expr._ir, keep)))
        return cleanup(m)

    def unfilter_entries(self):
        """Unfilters filtered entries, populating fields with missing values.

        Returns
        -------
        :class:`MatrixTable`

        Notes
        -----
        This method is used in the case that a pipeline downstream of :meth:`filter_entries`
        requires a fully dense (no filtered entries) matrix table.

        Generally, if this method is required in a pipeline, the upstream pipeline can
        be rewritten to use annotation instead of entry filtering.

        See Also
        --------
        :meth:`filter_entries`, :meth:`compute_entry_filter_stats`
        """
        entry_ir = hl.if_else(
            hl.is_defined(self.entry),
            self.entry,
            hl.struct(**{k: hl.missing(v.dtype) for k, v in self.entry.items()}))._ir
        return MatrixTable(ir.MatrixMapEntries(self._mir, entry_ir))

    @typecheck_method(row_field=str, col_field=str)
    def compute_entry_filter_stats(self, row_field='entry_stats_row', col_field='entry_stats_col') -> 'MatrixTable':
        """Compute statistics about the number and fraction of filtered entries.

        .. include:: _templates/experimental.rst

        Parameters
        ----------
        row_field : :class:`str`
            Name for computed row field (default: ``entry_stats_row``.
        col_field : :class:`str`
            Name for computed column field (default: ``entry_stats_col``.

        Returns
        -------
        :class:`.MatrixTable`

        Notes
        -----
        Adds a new row field, `row_field`, and a new column field, `col_field`,
        each of which are structs with the following fields:

         - *n_filtered* (:data:`.tint64`) - Number of filtered entries per row
           or column.
         - *n_remaining* (:data:`.tint64`) - Number of entries not filtered per
           row or column.
         - *fraction_filtered* (:data:`.tfloat32`) - Number of filtered entries
           divided by the total number of filtered and remaining entries.

        See Also
        --------
        :meth:`filter_entries`, :meth:`unfilter_entries`
        """
        def result(count):
            return hl.rbind(count,
                            hl.agg.count(),
                            lambda n_tot, n_def: hl.struct(n_filtered=n_tot - n_def,
                                                           n_remaining=n_def,
                                                           fraction_filtered=(n_tot - n_def) / n_tot))
        mt = self
        mt = mt.annotate_cols(**{col_field: result(mt.count_rows(_localize=False))})
        mt = mt.annotate_rows(**{row_field: result(mt.count_cols(_localize=False))})
        return mt

    @typecheck_method(named_exprs=expr_any)
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
        check_annotate_exprs(caller, named_exprs, self._global_indices, set())
        fields_referenced = extract_refs_by_indices(named_exprs.values(), self._global_indices) - set(named_exprs.keys())
        return self._select_globals(caller,
                                    self.globals.annotate(**named_exprs).drop(*fields_referenced))

    @typecheck_method(named_exprs=expr_any)
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
        check_annotate_exprs(caller, named_exprs, self._row_indices, {self._col_axis})
        fields_referenced = extract_refs_by_indices(named_exprs.values(), self._row_indices) - set(named_exprs.keys())
        fields_referenced -= set(self.row_key)

        return self._select_rows(caller, self.row.annotate(**named_exprs).drop(*fields_referenced))

    @typecheck_method(named_exprs=expr_any)
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
        check_annotate_exprs(caller, named_exprs, self._col_indices, {self._row_axis})
        fields_referenced = extract_refs_by_indices(named_exprs.values(), self._col_indices) - set(named_exprs.keys())
        fields_referenced -= set(self.col_key)

        return self._select_cols(caller,
                                 self.col.annotate(**named_exprs).drop(*fields_referenced))

    @typecheck_method(named_exprs=expr_any)
    def transmute_entries(self, **named_exprs) -> 'MatrixTable':
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
        check_annotate_exprs(caller, named_exprs, self._entry_indices, set())
        fields_referenced = extract_refs_by_indices(named_exprs.values(), self._entry_indices) - set(named_exprs.keys())

        return self._select_entries(caller,
                                    self.entry.annotate(**named_exprs).drop(*fields_referenced))

    @typecheck_method(expr=expr_any, _localize=bool)
    def aggregate_rows(self, expr, _localize=True) -> Any:
        """Aggregate over rows to a local value.

        Examples
        --------
        Aggregate over rows:

        >>> dataset.aggregate_rows(hl.struct(n_high_quality=hl.agg.count_where(dataset.qual > 40),
        ...                                  mean_qual=hl.agg.mean(dataset.qual)))
        Struct(n_high_quality=9, mean_qual=140054.73333333334)

        Notes
        -----
        Unlike most :class:`.MatrixTable` methods, this method does not support
        meaningful references to fields that are not global or indexed by row.

        This method should be thought of as a more convenient alternative to
        the following:

        >>> rows_table = dataset.rows()
        >>> rows_table.aggregate(hl.struct(n_high_quality=hl.agg.count_where(rows_table.qual > 40),
        ...                                mean_qual=hl.agg.mean(rows_table.qual)))

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
        rows_table = ir.MatrixRowsTable(base._mir)
        subst_query = ir.subst(expr._ir, {}, {'va': ir.Ref('row', rows_table.typ.row_type)})

        agg_ir = ir.TableAggregate(rows_table, subst_query)
        if _localize:
            return Env.backend().execute(ir.MakeTuple([agg_ir]))[0]
        else:
            return construct_expr(ir.LiftMeOut(agg_ir), expr.dtype)

    @typecheck_method(expr=expr_any, _localize=bool)
    def aggregate_cols(self, expr, _localize=True) -> Any:
        """Aggregate over columns to a local value.

        Examples
        --------
        Aggregate over columns:

        >>> dataset.aggregate_cols(
        ...    hl.struct(fraction_female=hl.agg.fraction(dataset.pheno.is_female),
        ...              case_ratio=hl.agg.count_where(dataset.is_case) / hl.agg.count()))
        Struct(fraction_female=0.44, case_ratio=1.0)

        Notes
        -----
        Unlike most :class:`.MatrixTable` methods, this method does not support
        meaningful references to fields that are not global or indexed by column.

        This method should be thought of as a more convenient alternative to
        the following:

        >>> cols_table = dataset.cols()
        >>> cols_table.aggregate(
        ...     hl.struct(fraction_female=hl.agg.fraction(cols_table.pheno.is_female),
        ...               case_ratio=hl.agg.count_where(cols_table.is_case) / hl.agg.count()))

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

        cols_field = Env.get_uid()
        globals = base.localize_entries(columns_array_field_name=cols_field).index_globals()
        if len(self._col_key) == 0:
            cols = globals[cols_field]
        else:
            if Env.hc()._warn_cols_order:
                warning("aggregate_cols(): Aggregates over cols ordered by 'col_key'."
                        "\n    To preserve matrix table column order, "
                        "first unkey columns with 'key_cols_by()'")
                Env.hc()._warn_cols_order = False
            cols = hl.sorted(globals[cols_field], key=lambda x: x.select(*self._col_key.keys()))

        agg_ir = ir.Let(
            'global',
            globals.drop(cols_field)._ir,
            ir.StreamAgg(ir.ToStream(cols._ir), 'sa', expr._ir))

        if _localize:
            return Env.backend().execute(ir.MakeTuple([agg_ir]))[0]
        else:
            return construct_expr(agg_ir, expr.dtype)

    @typecheck_method(expr=expr_any, _localize=bool)
    def aggregate_entries(self, expr, _localize=True):
        """Aggregate over entries to a local value.

        Examples
        --------
        Aggregate over entries:

        >>> dataset.aggregate_entries(hl.struct(global_gq_mean=hl.agg.mean(dataset.GQ),
        ...                                     call_rate=hl.agg.fraction(hl.is_defined(dataset.GT))))
        Struct(global_gq_mean=69.60514541387025, call_rate=0.9933333333333333)

        Notes
        -----
        This method should be thought of as a more convenient alternative to
        the following:

        >>> entries_table = dataset.entries()
        >>> entries_table.aggregate(hl.struct(global_gq_mean=hl.agg.mean(entries_table.GQ),
        ...                                   call_rate=hl.agg.fraction(hl.is_defined(entries_table.GT))))

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
        agg_ir = ir.MatrixAggregate(base._mir, expr._ir)
        if _localize:
            return Env.backend().execute(ir.MakeTuple([agg_ir]))[0]
        else:
            return construct_expr(ir.LiftMeOut(agg_ir), expr.dtype)

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
        :class:`.MatrixTable`
            Matrix table exploded row-wise for each element of `field_expr`.
        """
        if isinstance(field_expr, str):
            if field_expr not in self._fields:
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
            while isinstance(nested, ir.GetField):
                root.append(nested.name)
                nested = nested.o
            root = root[::-1]

        if not isinstance(field_expr.dtype, (tarray, tset)):
            raise ValueError(f"method 'explode_rows' expects array or set, found: {field_expr.dtype}")

        if self.row_key is not None:
            for k in self.row_key.values():
                if k is field_expr:
                    raise ValueError("method 'explode_rows' cannot explode a key field")

        return MatrixTable(ir.MatrixExplodeRows(self._mir, root))

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
            if field_expr not in self._fields:
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
            while isinstance(nested, ir.GetField):
                root.append(nested.name)
                nested = nested.o
            root = root[::-1]

        if not isinstance(field_expr.dtype, (tarray, tset)):
            raise ValueError(f"method 'explode_cols' expects array or set, found: {field_expr.dtype}")

        if self.col_key is not None:
            for k in self.col_key.values():
                if k is field_expr:
                    raise ValueError("method 'explode_cols' cannot explode a key field")

        return MatrixTable(ir.MatrixExplodeCols(self._mir, root))

    @typecheck_method(exprs=oneof(str, Expression), named_exprs=expr_any)
    def group_rows_by(self, *exprs, **named_exprs) -> 'GroupedMatrixTable':
        """Group rows, used with :meth:`.GroupedMatrixTable.aggregate`.

        Examples
        --------
        Aggregate to a matrix with genes as row keys, computing the number of
        non-reference calls as an entry field:

        >>> dataset_result = (dataset.group_rows_by(dataset.gene)
        ...                          .aggregate(n_non_ref = hl.agg.count_where(dataset.GT.is_non_ref())))

        Notes
        -----
        All complex expressions must be passed as named expressions.

        Parameters
        ----------
        exprs : args of :class:`str` or :class:`.Expression`
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
        ...                          .aggregate(call_rate = hl.agg.fraction(hl.is_defined(dataset.GT))))

        Notes
        -----
        All complex expressions must be passed as named expressions.

        Parameters
        ----------
        exprs : args of :class:`str` or :class:`.Expression`
            Column fields to group by.
        named_exprs : keyword args of :class:`.Expression`
            Column-indexed expressions to group by.

        Returns
        -------
        :class:`.GroupedMatrixTable`
            Grouped matrix, can be used to call :meth:`.GroupedMatrixTable.aggregate`.
        """
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

        >>> mt.cols().show() # doctest: +SKIP_OUTPUT_CHECK
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

        >>> mt.entries().show() # doctest: +SKIP_OUTPUT_CHECK
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

        >>> mt.entries().show() # doctest: +SKIP_OUTPUT_CHECK
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

        return MatrixTable(ir.MatrixCollectColsByKey(self._mir))

    @typecheck_method(_localize=bool)
    def count_rows(self, _localize=True) -> int:
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
        count_ir = ir.TableCount(ir.MatrixRowsTable(self._mir))
        if _localize:
            return Env.backend().execute(count_ir)
        else:
            return construct_expr(ir.LiftMeOut(count_ir), hl.tint64)

    def _force_count_rows(self):
        return Env.backend().execute(ir.MatrixToValueApply(self._mir, {'name': 'ForceCountMatrixTable'}))

    def _force_count_cols(self):
        return self.cols()._force_count()

    @typecheck_method(_localize=bool)
    def count_cols(self, _localize=True) -> int:
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
        count_ir = ir.TableCount(ir.MatrixColsTable(self._mir))
        if _localize:
            return Env.backend().execute(count_ir)
        else:
            return construct_expr(ir.LiftMeOut(count_ir), hl.tint64)

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
        count_ir = ir.MatrixCount(self._mir)
        return Env.backend().execute(count_ir)

    @typecheck_method(output=str,
                      overwrite=bool,
                      stage_locally=bool,
                      _codec_spec=nullable(str),
                      _read_if_exists=bool,
                      _intervals=nullable(sequenceof(anytype)),
                      _filter_intervals=bool,
                      _drop_cols=bool,
                      _drop_rows=bool)
    def checkpoint(self, output: str, overwrite: bool = False, stage_locally: bool = False,
                   _codec_spec: Optional[str] = None, _read_if_exists: bool = False,
                   _intervals=None, _filter_intervals=False, _drop_cols=False, _drop_rows=False) -> 'MatrixTable':
        """Checkpoint the matrix table to disk by writing and reading using a fast, but less space-efficient codec.

        Parameters
        ----------
        output : str
            Path at which to write.
        stage_locally: bool
            If ``True``, major output will be written to temporary local storage
            before being copied to ``output``
        overwrite : bool
            If ``True``, overwrite an existing file at the destination.

        Returns
        -------
        :class:`MatrixTable`


        .. include:: _templates/write_warning.rst

        Notes
        -----
        An alias for :meth:`write` followed by :func:`.read_matrix_table`. It is
        possible to read the file at this path later with
        :func:`.read_matrix_table`. A faster, but less efficient, codec is used
        or writing the data so the file will be larger than if one used
        :meth:`write`.

        Examples
        --------
        >>> dataset = dataset.checkpoint('output/dataset_checkpoint.mt')
        """
        hl.current_backend().validate_file(output)

        if not _read_if_exists or not hl.hadoop_exists(f'{output}/_SUCCESS'):
            self.write(output=output, overwrite=overwrite, stage_locally=stage_locally, _codec_spec=_codec_spec)
            _assert_type = self._type
            _load_refs = False
        else:
            _assert_type = None
            _load_refs = True
        return hl.read_matrix_table(
            output,
            _intervals=_intervals,
            _filter_intervals=_filter_intervals,
            _drop_cols=_drop_cols,
            _drop_rows=_drop_rows,
            _assert_type=_assert_type,
            _load_refs=_load_refs
        )

    @typecheck_method(output=str,
                      overwrite=bool,
                      stage_locally=bool,
                      _codec_spec=nullable(str),
                      _partitions=nullable(expr_any))
    def write(self, output: str, overwrite: bool = False, stage_locally: bool = False,
              _codec_spec: Optional[str] = None, _partitions=None):
        """Write to disk.

        Examples
        --------

        >>> dataset.write('output/dataset.mt')

        .. include:: _templates/write_warning.rst

        See Also
        --------
        :func:`.read_matrix_table`

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

        hl.current_backend().validate_file(output)

        if _partitions is not None:
            _partitions, _partitions_type = hl.utils._dumps_partitions(_partitions, self.row_key.dtype)
        else:
            _partitions_type = None

        writer = ir.MatrixNativeWriter(output, overwrite, stage_locally, _codec_spec, _partitions, _partitions_type)
        Env.backend().execute(ir.MatrixWrite(self._mir, writer))

    class _Show:
        def __init__(self, table, n_rows, actual_n_cols, displayed_n_cols, width, truncate, types):
            self.table_show = table._show(n_rows, width, truncate, types)
            self.actual_n_cols = actual_n_cols
            self.displayed_n_cols = displayed_n_cols

        def __str__(self):
            s = self.table_show.__str__()
            if self.displayed_n_cols != self.actual_n_cols:
                s += f"showing the first { self.displayed_n_cols } of { self.actual_n_cols } columns"
            return s

        def __repr__(self):
            return self.__str__()

        def _repr_html_(self):
            s = self.table_show._repr_html_()
            if self.displayed_n_cols != self.actual_n_cols:
                s += '<p style="background: #fdd; padding: 0.4em;">'
                s += f"showing the first { self.displayed_n_cols } of { self.actual_n_cols } columns"
                s += '</p>\n'
            return s

    @typecheck_method(n_rows=nullable(int),
                      n_cols=nullable(int),
                      include_row_fields=bool,
                      width=nullable(int),
                      truncate=nullable(int),
                      types=bool,
                      handler=nullable(anyfunc))
    def show(self,
             n_rows=None,
             n_cols=None,
             include_row_fields=False,
             width=None,
             truncate=None,
             types=True,
             handler=None):
        """Print the first few rows of the matrix table to the console.

        .. include:: _templates/experimental.rst

        Notes
        -----
        The output can be passed piped to another output source using the `handler` argument:

        >>> mt.show(handler=lambda x: logging.info(x))  # doctest: +SKIP

        Parameters
        ----------
        n_rows : :obj:`int`
            Maximum number of rows to show.
        n_cols : :obj:`int`
            Maximum number of columns to show.
        width : :obj:`int`
            Horizontal width at which to break fields.
        truncate : :obj:`int`, optional
            Truncate each field to the given number of characters. If
            ``None``, truncate fields to the given `width`.
        types : :obj:`bool`
            Print an extra header line with the type of each field.
        handler : Callable[[str], Any]
            Handler function for data string.
        """

        def estimate_size(struct_expression):
            return sum(max(len(f), len(str(x.dtype))) + 3
                       for f, x in struct_expression.flatten().items())

        if n_cols is None:
            import shutil
            (characters, _) = shutil.get_terminal_size((80, 10))
            characters -= 6  # borders
            key_characters = estimate_size(self.row_key)
            characters -= key_characters
            if include_row_fields:
                characters -= estimate_size(self.row_value)
            characters = max(characters, 0)
            n_cols = characters // (estimate_size(self.entry) + 4)  # 4 for the column index
        actual_n_cols = self.count_cols()
        displayed_n_cols = min(actual_n_cols, n_cols)

        t = self.localize_entries('entries', 'cols')
        if len(t.key) > 0:
            t = t.order_by(*t.key)
        col_key_type = self.col_key.dtype

        col_headers = [f'<col {i}>' for i in range(0, displayed_n_cols)]
        if len(col_key_type) == 1 and col_key_type[0] in (hl.tstr, hl.tint32, hl.tint64):
            cols = self.col_key[0].take(displayed_n_cols)
            if len(set(cols)) == len(cols):
                col_headers = [repr(c) for c in cols]

        entries = {col_headers[i]: t.entries[i]
                   for i in range(0, displayed_n_cols)}
        t = t.select(
            **{f: t[f] for f in self.row_key},
            **{f: t[f] for f in self.row_value if include_row_fields},
            **entries)
        if handler is None:
            handler = default_handler()
        return handler(MatrixTable._Show(t, n_rows, actual_n_cols, displayed_n_cols, width, truncate, types))

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
        return Table.parallelize(
            [hl.eval(self.globals)], self._global_type)

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

        return Table(ir.MatrixRowsTable(self._mir))

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
            warning("cols(): Resulting column table is sorted by 'col_key'."
                    "\n    To preserve matrix table column order, "
                    "first unkey columns with 'key_cols_by()'")
            Env.hc()._warn_cols_order = False

        return Table(ir.MatrixColsTable(self._mir))

    def entries(self) -> Table:
        """Returns a matrix in coordinate table form.

        Examples
        --------
        Extract the entry table:

        >>> entries_table = dataset.entries()

        Notes
        -----
        The coordinate table representation of the source matrix table contains
        one row for each **non-filtered** entry of the matrix -- if a matrix table
        has no filtered entries and contains N rows and M columns, the table will contain
        ``M * N`` rows, which can be **a very large number**.

        This representation can be useful for aggregating over both axes of a matrix table
        at the same time -- it is not possible to aggregate over a matrix table using
        :meth:`group_rows_by` and :meth:`group_cols_by` at the same time (aggregating
        by population and chromosome from a variant-by-sample genetics representation,
        for instance). After moving to the coordinate representation with :meth:`entries`,
        it is possible to group and aggregate the resulting table much more flexibly,
        albeit with potentially poorer computational performance.

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

        Warning
        -------
        If the matrix table has no row key, but has a column key, this operation
        may require a full shuffle to sort by the column key, depending on the
        pipeline.

        Returns
        -------
        :class:`.Table`
            Table with all non-global fields from the matrix, with **one row per entry of the matrix**.
        """
        if Env.hc()._warn_entries_order and len(self.col_key) > 0:
            warning("entries(): Resulting entries table is sorted by '(row_key, col_key)'."
                    "\n    To preserve row-major matrix table order, "
                    "first unkey columns with 'key_cols_by()'")
            Env.hc()._warn_entries_order = False

        return Table(ir.MatrixEntriesTable(self._mir))

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
        return construct_expr(ir.TableGetGlobals(ir.MatrixRowsTable(self._mir)), self.globals.dtype)

    def index_rows(self, *exprs, all_matches=False) -> 'Expression':
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
        all_matches : bool
            Experimental. If ``True``, value of expression is array of all matches.

        Notes
        -----
        ``index_rows(exprs)`` is equivalent to ``rows().index(exprs)``
        or ``rows()[exprs]``.

        The type of the resulting struct is the same as the type of
        :meth:`.row_value`.

        Returns
        -------
        :class:`.Expression`
        """
        try:
            return self.rows()._index(*exprs, all_matches=all_matches)
        except TableIndexKeyError as err:
            raise ExpressionException(
                f"Key type mismatch: cannot index matrix table with given expressions:\n"
                f"  MatrixTable row key: {', '.join(str(t) for t in err.key_type.values()) or '<<<empty key>>>'}\n"
                f"  Index expressions:   {', '.join(str(e.dtype) for e in err.index_expressions)}")

    def index_cols(self, *exprs, all_matches=False) -> 'Expression':
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
        all_matches : bool
            Experimental. If ``True``, value of expression is array of all matches.

        Notes
        -----
        ``index_cols(cols)`` is equivalent to ``cols().index(exprs)``
        or ``cols()[exprs]``.

        The type of the resulting struct is the same as the type of
        :meth:`.col_value`.

        Returns
        -------
        :class:`.Expression`
        """
        try:
            return self.cols()._index(*exprs, all_matches=all_matches)
        except TableIndexKeyError as err:
            raise ExpressionException(
                f"Key type mismatch: cannot index matrix table with given expressions:\n"
                f"  MatrixTable col key: {', '.join(str(t) for t in err.key_type.values()) or '<<<empty key>>>'}\n"
                f"  Index expressions:   {', '.join(str(e.dtype) for e in err.index_expressions)}")

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
        if len(row_exprs) == 0 or len(col_exprs) == 0:
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
                    f"Key type mismatch: Cannot index matrix table with given expressions\n"
                    f"  MatrixTable row key:   {', '.join(str(t) for t in self.row_key.dtype.values())}\n"
                    f"  Row index expressions: {', '.join(str(e.dtype) for e in row_exprs)}")

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
                    f"  MatrixTable col key:   {', '.join(str(t) for t in self.col_key.dtype.values())}\n"
                    f"  Col index expressions: {', '.join(str(e.dtype) for e in col_exprs)}")

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
                left = left._annotate_all(row_exprs={row_uid: localized.index(*row_exprs)[row_uid]},
                                          col_exprs={col_uid: src_cols_indexed.index(*col_exprs)[col_uid]})
                return left.annotate_entries(**{uid: left[row_uid][left[col_uid]]})

            join_ir = ir.Join(ir.ProjectedTopLevelReference('g', uid, self.entry.dtype),
                              uids,
                              [*row_exprs, *col_exprs],
                              joiner)
            return construct_expr(join_ir, self.entry.dtype, indices, aggregations)

    @typecheck_method(entries_field_name=str, cols_field_name=str)
    def _localize_entries(self, entries_field_name, cols_field_name) -> 'Table':
        assert entries_field_name not in self.row
        assert cols_field_name not in self.globals
        return Table(ir.CastMatrixToTable(
            self._mir, entries_field_name, cols_field_name))

    @typecheck_method(entries_array_field_name=nullable(str),
                      columns_array_field_name=nullable(str))
    def localize_entries(self,
                         entries_array_field_name=None,
                         columns_array_field_name=None) -> 'Table':
        """Convert the matrix table to a table with entries localized as an array of structs.

        Examples
        --------
        Build a numpy ndarray from a small :class:`.MatrixTable`:

        >>> mt = hl.utils.range_matrix_table(3,3)
        >>> mt = mt.select_entries(x = mt.row_idx * mt.col_idx)
        >>> mt.show()
        +---------+-------+-------+-------+
        | row_idx |   0.x |   1.x |   2.x |
        +---------+-------+-------+-------+
        |   int32 | int32 | int32 | int32 |
        +---------+-------+-------+-------+
        |       0 |     0 |     0 |     0 |
        |       1 |     0 |     1 |     2 |
        |       2 |     0 |     2 |     4 |
        +---------+-------+-------+-------+

        >>> t = mt.localize_entries('entry_structs', 'columns')
        >>> t.describe()
        ----------------------------------------
        Global fields:
            'columns': array<struct {
                col_idx: int32
            }>
        ----------------------------------------
        Row fields:
            'row_idx': int32
            'entry_structs': array<struct {
                x: int32
            }>
        ----------------------------------------
        Key: ['row_idx']
        ----------------------------------------

        >>> t = t.select(entries = t.entry_structs.map(lambda entry: entry.x))
        >>> import numpy as np
        >>> np.array(t.entries.collect())
        array([[0, 0, 0],
               [0, 1, 2],
               [0, 2, 4]])

        Notes
        -----
        Both of the added fields are arrays of length equal to
        ``mt.count_cols()``. Missing entries are represented as missing structs
        in the entries array.

        Parameters
        ----------
        entries_array_field_name : :class:`str`
            The name of the table field containing the array of entry structs
            for the given row.
        columns_array_field_name : :class:`str`
            The name of the global field containing the array of column
            structs.

        Returns
        -------
        :class:`.Table`
            A table whose fields are the row fields of this matrix table plus
            one field named ``entries_array_field_name``. The global fields of
            this table are the global fields of this matrix table plus one field
            named ``columns_array_field_name``.
        """
        entries = entries_array_field_name or Env.get_uid()
        cols = columns_array_field_name or Env.get_uid()
        if entries in self.row:
            raise ValueError(
                f"'localize_entries': cannot localize entries to field {entries!r}, which is already a row field")
        if cols in self.globals:
            raise ValueError(
                f"'localize_entries': cannot localize columns to field {cols!r}, which is already a global field")

        t = self._localize_entries(entries, cols)
        if entries_array_field_name is None:
            t = t.drop(entries)
        if columns_array_field_name is None:
            t = t.drop(cols)
        return t

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

        for field_name in list(itertools.chain(row_exprs.keys(),
                                               col_exprs.keys(),
                                               entry_exprs.keys(),
                                               global_exprs.keys())):
            if field_name in self._fields:
                raise RuntimeError(f'field {repr(field_name)} already in matrix table, cannot use _annotate_all')

        base, cleanup = self._process_joins(*all_exprs)
        mir = base._mir

        if row_exprs:
            row_struct = ir.InsertFields.construct_with_deduplication(
                base.row._ir, [(n, e._ir) for (n, e) in row_exprs.items()], None)
            mir = ir.MatrixMapRows(mir, row_struct)
        if col_exprs:
            col_struct = ir.InsertFields.construct_with_deduplication(
                base.col._ir, [(n, e._ir) for (n, e) in col_exprs.items()], None)
            mir = ir.MatrixMapCols(mir, col_struct, None)
        if entry_exprs:
            entry_struct = ir.InsertFields.construct_with_deduplication(
                base.entry._ir, [(n, e._ir) for (n, e) in entry_exprs.items()], None)
            mir = ir.MatrixMapEntries(mir, entry_struct)
        if global_exprs:
            globals_struct = ir.InsertFields.construct_with_deduplication(
                base.globals._ir, [(n, e._ir) for (n, e) in global_exprs.items()], None)
            mir = ir.MatrixMapGlobals(mir, globals_struct)

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

        all_names = list(itertools.chain(row_exprs.keys(),
                                         col_exprs.keys(),
                                         entry_exprs.keys(),
                                         global_exprs.keys()))
        uids = {k: Env.get_uid() for k in all_names}

        mt = self._annotate_all({uids[k]: v for k, v in row_exprs.items()},
                                {uids[k]: v for k, v in col_exprs.items()},
                                {uids[k]: v for k, v in entry_exprs.items()},
                                {uids[k]: v for k, v in global_exprs.items()})

        keep = set()
        if row_key is not None:
            old_key = list(mt.row_key)
            mt = mt.key_rows_by(*(uids[k] for k in row_key)).drop(*old_key)
        else:
            keep = keep.union(set(mt.row_key))

        if col_key is not None:
            old_key = list(mt.col_key)
            mt = mt.key_cols_by(*(uids[k] for k in col_key)).drop(*old_key)
        else:
            keep = keep.union(set(mt.col_key))

        keep = keep.union(uids.values())
        return (mt.drop(*(f for f in mt._fields if f not in keep))
                .rename({uid: original for original, uid in uids.items()}))

    def _process_joins(self, *exprs) -> 'MatrixTable':
        return process_joins(self, exprs)

    def describe(self, handler=print, *, widget=False):
        """Print information about the fields in the matrix table.

        Note
        ----
        The `widget` argument is **experimental**.

        Parameters
        ----------
        handler : Callable[[str], None]
            Handler function for returned string.
        widget : bool
            Create an interactive IPython widget.
        """
        if widget:
            from hail.experimental.interact import interact
            return interact(self)

        def format_type(typ):
            return typ.pretty(indent=4).lstrip()

        if len(self.globals) == 0:
            global_fields = '\n    None'
        else:
            global_fields = ''.join("\n    '{name}': {type}".format(
                name=f, type=format_type(t)) for f, t in self.globals.dtype.items())

        if len(self.row) == 0:
            row_fields = '\n    None'
        else:
            row_fields = ''.join("\n    '{name}': {type}".format(
                name=f, type=format_type(t)) for f, t in self.row.dtype.items())

        row_key = '[' + ', '.join("'{name}'".format(name=f) for f in self.row_key) + ']' \
            if self.row_key else None

        if len(self.col) == 0:
            col_fields = '\n    None'
        else:
            col_fields = ''.join("\n    '{name}': {type}".format(
                name=f, type=format_type(t)) for f, t in self.col.dtype.items())

        col_key = '[' + ', '.join("'{name}'".format(name=f) for f in self.col_key) + ']' \
            if self.col_key else None

        if len(self.entry) == 0:
            entry_fields = '\n    None'
        else:
            entry_fields = ''.join("\n    '{name}': {type}".format(
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
        return MatrixTable(ir.MatrixChooseCols(self._mir, indices))

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
        return Env.backend().execute(ir.MatrixToValueApply(self._mir, {'name': 'NPartitionsMatrixTable'}))

    @typecheck_method(n_partitions=int,
                      shuffle=bool)
    def repartition(self, n_partitions: int, shuffle: bool = True) -> 'MatrixTable':
        """Change the number of partitions.

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
        for details.

        When ``shuffle=True``, Hail does a full shuffle of the data
        and creates equal sized partitions.  When ``shuffle=False``,
        Hail combines existing partitions to avoid a full
        shuffle. These algorithms correspond to the `repartition` and
        `coalesce` commands in Spark, respectively. In particular,
        when ``shuffle=False``, ``n_partitions`` cannot exceed current
        number of partitions.

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
        if hl.current_backend().requires_lowering:
            tmp = hl.utils.new_temp_file()

            if len(self.row_key) == 0:
                uid = Env.get_uid()
                tmp2 = hl.utils.new_temp_file()
                self.checkpoint(tmp2)
                ht = hl.read_matrix_table(tmp2).add_row_index(uid).key_rows_by(uid)
                ht.checkpoint(tmp)
                return hl.read_matrix_table(tmp, _n_partitions=n_partitions).drop(uid)
            else:
                # checkpoint rather than write to use fast codec
                self.checkpoint(tmp)
                return hl.read_matrix_table(tmp, _n_partitions=n_partitions)

        return MatrixTable(ir.MatrixRepartition(
            self._mir, n_partitions,
            ir.RepartitionStrategy.SHUFFLE if shuffle else ir.RepartitionStrategy.COALESCE))

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
        return MatrixTable(ir.MatrixRepartition(
            self._mir, max_partitions, ir.RepartitionStrategy.NAIVE_COALESCE))

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
        return Env.backend().persist(self)

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
        return Env.backend().unpersist(self)

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
        name : :class:`str`
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
        name: :class:`str`
            Name for column index field.

        Returns
        -------
        :class:`.MatrixTable`
            Dataset with new field.
        """
        return self.annotate_cols(**{name: hl.scan.count()})

    @typecheck_method(other=matrix_table_type,
                      tolerance=numeric,
                      absolute=bool,
                      reorder_fields=bool)
    def _same(self, other, tolerance=1e-6, absolute=False, reorder_fields=False) -> bool:
        entries_name = Env.get_uid('entries_')
        cols_name = Env.get_uid('columns_')

        fd_f = set if reorder_fields else list

        if fd_f(self.row) != fd_f(other.row):
            print(f'Different row fields: \n  {list(self.row)}\n  {list(other.row)}')
            return False
        if fd_f(self.globals) != fd_f(other.globals):
            print(f'Different globals fields: \n  {list(self.globals)}\n  {list(other.globals)}')
            return False
        if fd_f(self.col) != fd_f(other.col):
            print(f'Different col fields: \n  {list(self.col)}\n  {list(other.col)}')
            return False
        if fd_f(self.entry) != fd_f(other.entry):
            print(f'Different row fields: \n  {list(self.entry)}\n  {list(other.entry)}')
            return False

        if reorder_fields:
            entry_order = list(self.entry)
            if list(other.entry) != entry_order:
                other = other.select_entries(*entry_order)

            globals_order = list(self.globals)
            if list(other.globals) != globals_order:
                other = other.select_globals(*globals_order)

            col_order = list(self.col)
            if list(other.col) != col_order:
                other = other.select_cols(*col_order)

            row_order = list(self.row)
            if list(other.row) != row_order:
                other = other.select_rows(*row_order)

        if list(self.col_key) != list(other.col_key):
            print(f'different col keys:\n  {list(self.col_key)}\n  {list(other.col_key)}')
            return False

        return self._localize_entries(entries_name, cols_name)._same(
            other._localize_entries(entries_name, cols_name), tolerance, absolute)

    @typecheck_method(caller=str, s=expr_struct())
    def _select_entries(self, caller, s) -> 'MatrixTable':
        base, cleanup = self._process_joins(s)
        analyze(caller, s, self._entry_indices)
        return cleanup(MatrixTable(ir.MatrixMapEntries(base._mir, s._ir)))

    @typecheck_method(caller=str,
                      row=expr_struct())
    def _select_rows(self, caller, row) -> 'MatrixTable':
        analyze(caller, row, self._row_indices, {self._col_axis})
        base, cleanup = self._process_joins(row)
        return cleanup(MatrixTable(ir.MatrixMapRows(base._mir, row._ir)))

    @typecheck_method(caller=str,
                      col=expr_struct(),
                      new_key=nullable(sequenceof(str)))
    def _select_cols(self, caller, col, new_key=None) -> 'MatrixTable':
        analyze(caller, col, self._col_indices, {self._row_axis})
        base, cleanup = self._process_joins(col)
        return cleanup(MatrixTable(ir.MatrixMapCols(base._mir, col._ir, new_key)))

    @typecheck_method(caller=str, s=expr_struct())
    def _select_globals(self, caller, s) -> 'MatrixTable':
        base, cleanup = self._process_joins(s)
        analyze(caller, s, self._global_indices)
        return cleanup(MatrixTable(ir.MatrixMapGlobals(base._mir, s._ir)))

    @typecheck(datasets=matrix_table_type, _check_cols=bool)
    def union_rows(*datasets: 'MatrixTable', _check_cols=True) -> 'MatrixTable':
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
                        "row keys", 0, first.row_key.keys(), i + 1, next.row_key.keys()
                    ))
                if first.row.dtype != next.row.dtype:
                    raise ValueError(error_msg.format(
                        "row types", 0, first.row.dtype, i + 1, next.row.dtype
                    ))
                if first.entry.dtype != next.entry.dtype:
                    raise ValueError(error_msg.format(
                        "entry field types", 0, first.entry.dtype, i + 1, next.entry.dtype
                    ))
                if first.col_key.dtype != next.col_key.dtype:
                    raise ValueError(error_msg.format(
                        "col key types", 0, first.col_key.dtype, i + 1, next.col_key.dtype
                    ))
            if _check_cols:
                wrong_keys = hl.eval(hl.rbind(first.col_key.collect(_localize=False), lambda first_keys: (
                    hl.enumerate([mt.col_key.collect(_localize=False) for mt in datasets[1:]])
                    .find(lambda x: ~(x[1] == first_keys))[0])))
                if wrong_keys is not None:
                    raise ValueError(f"'MatrixTable.union_rows' expects all datasets to have the same columns. "
                                     f"Datasets 0 and {wrong_keys + 1} have different columns (or possibly different order).")
            return MatrixTable(ir.MatrixUnionRows(*[d._mir for d in datasets]))

    @typecheck_method(other=matrix_table_type,
                      row_join_type=enumeration('inner', 'outer'),
                      drop_right_row_fields=bool)
    def union_cols(self, other: 'MatrixTable', row_join_type: str = 'inner', drop_right_row_fields: bool = True) -> 'MatrixTable':
        """Take the union of dataset columns.

        Warning
        -------

        This method does not preserve the global fields from the other matrix table.

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

        This method creates a :class:`.MatrixTable` which contains all columns
        from both input datasets. The set of rows included in the result is
        determined by the `row_join_type` parameter.

        - With the default value of ``'inner'``, an inner join is performed
          on rows, so that only rows whose row key exists in both input datasets
          are included. In this case, the entries for each row are the
          concatenation of all entries of the corresponding rows in the input
          datasets.
        - With `row_join_type` set to  ``'outer'``, an outer join is perfomed on
          rows, so that row keys which exist in only one input dataset are also
          included. For those rows, the entry fields for the columns coming
          from the other dataset will be missing.

        Only distinct row keys from each dataset are included (equivalent to
        calling :meth:`.distinct_by_row` on each dataset first).

        This method does not deduplicate; if a column key exists identically in
        two datasets, then it will be duplicated in the result.

        Parameters
        ----------
        other : :class:`.MatrixTable`
            Dataset to concatenate.
        row_join_type : :obj:`.str`
            If `outer`, perform an outer join on rows; if 'inner', perform an
            inner join. Default `inner`.
        drop_right_row_fields : :obj:`.bool`
            If true, non-key row fields of `other` are dropped. Otherwise,
            non-key row fields in the two datasets must have distinct names,
            and the result contains the union of the row fields.

        Returns
        -------
        :class:`.MatrixTable`
            Dataset with columns from both datasets.
        """
        if self.entry.dtype != other.entry.dtype:
            raise ValueError(f'entry types differ:\n'
                             f'    left: {self.entry.dtype}\n'
                             f'    right: {other.entry.dtype}')
        if self.col.dtype != other.col.dtype:
            raise ValueError(f'column types differ:\n'
                             f'    left: {self.col.dtype}\n'
                             f'    right: {other.col.dtype}')
        if self.col_key.keys() != other.col_key.keys():
            raise ValueError(f'column key fields differ:\n'
                             f'    left: {", ".join(self.col_key.keys())}\n'
                             f'    right: {", ".join(other.col_key.keys())}')
        if list(self.row_key.dtype.values()) != list(other.row_key.dtype.values()):
            raise ValueError(f'row key types differ:\n'
                             f'    left: {", ".join(self.row_key.dtype.values())}\n'
                             f'    right: {", ".join(other.row_key.dtype.values())}')

        if drop_right_row_fields:
            other = other.select_rows()
        else:
            left_fields = set(self.row_value)
            other_fields = set(other.row_value) - set(other.row_key)
            renames, _ = deduplicate(
                other_fields, max_attempts=100, already_used=left_fields)

            if renames:
                renames = dict(renames)
                other = other.rename(renames)
                info('Table.union_cols: renamed the following fields on the right to avoid name conflicts:'
                     + ''.join(f'\n    {repr(k)} -> {repr(v)}' for k, v in renames.items()))

        return MatrixTable(ir.MatrixUnionCols(self._mir, other._mir, row_join_type))

    @typecheck_method(n_rows=nullable(int), n_cols=nullable(int), n=nullable(int))
    def head(self, n_rows: Optional[int], n_cols: Optional[int] = None, *, n: Optional[int] = None) -> 'MatrixTable':
        """Subset matrix to first `n_rows` rows and `n_cols` cols.

        Examples
        --------
        >>> mt_range = hl.utils.range_matrix_table(100, 100)

        Passing only one argument will take the first `n_rows` rows:

        >>> mt_range.head(10).count()
        (10, 100)

        Passing two arguments refers to rows and columns, respectively:

        >>> mt_range.head(10, 20).count()
        (10, 20)

        Either argument may be ``None`` to indicate no filter.

        First 10 rows, all columns:

        >>> mt_range.head(10, None).count()
        (10, 100)

        All rows, first 10 columns:

        >>> mt_range.head(None, 10).count()
        (100, 10)

        Notes
        -----
        The number of partitions in the new matrix is equal to the number of
        partitions containing the first `n_rows` rows.

        Parameters
        ----------
        n_rows : :obj:`int`
            Number of rows to include (all rows included if ``None``).
        n_cols : :obj:`int`, optional
            Number of cols to include (all cols included if ``None``).
        n : :obj:`int`
            Deprecated in favor of n_rows.

        Returns
        -------
        :class:`.MatrixTable`
            Matrix including the first `n_rows` rows and first `n_cols` cols.

        """
        if n_rows is not None and n is not None:
            raise ValueError('Both n and n_rows specified. Only one may be specified.')

        if n_rows is not None:
            n_rows_name = 'n_rows'
        else:
            warnings.warn("MatrixTable.head: the 'n' parameter is deprecated in favor of 'n_rows'.")
            n_rows = n
            n_rows_name = 'n'
        del n

        mt = self
        if n_rows is not None:
            if n_rows < 0:
                raise ValueError(f"MatrixTable.head: expect '{n_rows_name}' to be non-negative or None, found '{n_rows}'")
            mt = MatrixTable(ir.MatrixRowsHead(mt._mir, n_rows))
        if n_cols is not None:
            if n_cols < 0:
                raise ValueError(f"MatrixTable.head: expect 'n_cols' to be non-negative or None, found '{n_cols}'")
            mt = MatrixTable(ir.MatrixColsHead(mt._mir, n_cols))
        return mt

    @typecheck_method(n_rows=nullable(int), n_cols=nullable(int), n=nullable(int))
    def tail(self, n_rows: Optional[int], n_cols: Optional[int] = None, *, n: Optional[int] = None) -> 'MatrixTable':
        """Subset matrix to last `n` rows.

        Examples
        --------
        >>> mt_range = hl.utils.range_matrix_table(100, 100)

        Passing only one argument will take the last `n` rows:

        >>> mt_range.tail(10).count()
        (10, 100)

        Passing two arguments refers to rows and columns, respectively:

        >>> mt_range.tail(10, 20).count()
        (10, 20)

        Either argument may be ``None`` to indicate no filter.

        Last 10 rows, all columns:

        >>> mt_range.tail(10, None).count()
        (10, 100)

        All rows, last 10 columns:

        >>> mt_range.tail(None, 10).count()
        (100, 10)

        Notes
        -----
        For backwards compatibility, the `n` parameter is not named `n_rows`,
        but the parameter refers to the number of rows to keep.

        The number of partitions in the new matrix is equal to the number of
        partitions containing the last `n` rows.

        Parameters
        ----------
        n_rows : :obj:`int`
            Number of rows to include (all rows included if ``None``).
        n_cols : :obj:`int`, optional
            Number of cols to include (all cols included if ``None``).
        n : :obj:`int`
            Deprecated in favor of n_rows.

        Returns
        -------
        :class:`.MatrixTable`
            Matrix including the last `n` rows and last `n_cols` cols.
        """
        if n_rows is not None and n is not None:
            raise ValueError('Both n and n_rows specified. Only one may be specified.')

        if n_rows is not None:
            n_rows_name = 'n_rows'
        else:
            warnings.warn("MatrixTable.tail: the 'n' parameter is deprecated in favor of 'n_rows'.")
            n_rows = n
            n_rows_name = 'n'
        del n

        mt = self
        if n_rows is not None:
            if n_rows < 0:
                raise ValueError(f"MatrixTable.tail: expect '{n_rows_name}' to be non-negative or None, found '{n_rows}'")
            mt = MatrixTable(ir.MatrixRowsTail(mt._mir, n_rows))
        if n_cols is not None:
            if n_cols < 0:
                raise ValueError(f"MatrixTable.tail: expect 'n_cols' to be non-negative or None, found '{n_cols}'")
            mt = MatrixTable(ir.MatrixColsTail(mt._mir, n_cols))
        return mt

    @typecheck_method(parts=sequenceof(int), keep=bool)
    def _filter_partitions(self, parts, keep=True) -> 'MatrixTable':
        return MatrixTable(ir.MatrixToMatrixApply(self._mir, {'name': 'MatrixFilterPartitions', 'parts': parts, 'keep': keep}))

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
        >>> sites_mt = hl.MatrixTable.from_rows_table(table)

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
        col_values_uid = Env.get_uid()
        entries_uid = Env.get_uid()
        return (table.annotate_globals(**{col_values_uid: hl.empty_array(hl.tstruct())})
                .annotate(**{entries_uid: hl.empty_array(hl.tstruct())})
                ._unlocalize_entries(entries_uid, col_values_uid, []))

    @typecheck_method(p=numeric,
                      seed=nullable(int))
    def sample_rows(self, p: float, seed=None) -> 'MatrixTable':
        """Downsample the matrix table by keeping each row with probability ``p``.

        Examples
        --------
        Downsample the dataset to approximately 1% of its rows.

        >>> small_dataset = dataset.sample_rows(0.01)

        Notes
        -----
        Although the :class:`MatrixTable` returned by this method may be
        small, it requires a full pass over the rows of the sampled object.

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

        if not 0 <= p <= 1:
            raise ValueError("Requires 'p' in [0,1]. Found p={}".format(p))

        return self.filter_rows(hl.rand_bool(p, seed))

    @typecheck_method(p=numeric,
                      seed=nullable(int))
    def sample_cols(self, p: float, seed=None) -> 'MatrixTable':
        """Downsample the matrix table by keeping each column with probability ``p``.

        Examples
        --------
        Downsample the dataset to approximately 1% of its columns.

        >>> small_dataset = dataset.sample_cols(0.01)

        Parameters
        ----------
        p : :obj:`float`
            Probability of keeping each column.
        seed : :obj:`int`
            Random seed.

        Returns
        -------
        :class:`.MatrixTable`
            Matrix table with approximately ``p * n_cols`` column.
        """

        if not 0 <= p <= 1:
            raise ValueError("Requires 'p' in [0,1]. Found p={}".format(p))

        return self.filter_cols(hl.rand_bool(p, seed))

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
        fields : :obj:`dict` from :class:`str` to :obj:`str`
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

        return MatrixTable(ir.MatrixRename(self._mir, global_map, col_map, row_map, entry_map))

    def distinct_by_row(self) -> 'MatrixTable':
        """Remove rows with a duplicate row key, keeping exactly one row for each unique key.

        Returns
        -------
        :class:`.MatrixTable`
        """
        return MatrixTable(ir.MatrixDistinctByRow(self._mir))

    def distinct_by_col(self) -> 'MatrixTable':
        """Remove columns with a duplicate row key, keeping exactly one column for each unique key.

        Returns
        -------
        :class:`.MatrixTable`
        """
        index_uid = Env.get_uid()

        col_key_fields = list(self.col_key)
        t = self.key_cols_by().cols()

        t = t.add_index(index_uid)
        unique_cols = t.aggregate(
            hl.agg.group_by(
                hl.struct(**{f: t[f] for f in col_key_fields}), hl.agg.take(t[index_uid], 1)))
        unique_cols = sorted([v[0] for _, v in unique_cols.items()])

        return self.choose_cols(unique_cols)

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
        :meth:`.make_table`:

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
        separator : :class:`str`
            Separator between sample IDs and entry field names.

        Returns
        -------
        :class:`.Table`

        """
        if not (len(self.col_key) == 1 and self.col_key[0].dtype == hl.tstr):
            raise ValueError("column key must be a single field of type str")

        col_keys = self.col_key[0].collect()

        counts = Counter(col_keys)
        if counts[None] > 0:
            raise ValueError("'make_table' encountered a missing column key; ensure all identifiers are defined.\n"
                             "  To fill in key index, run:\n"
                             "    mt = mt.key_cols_by(ck = hl.coalesce(mt.COL_KEY_NAME, 'missing_' + hl.str(hl.scan.count())))")

        duplicates = [k for k, count in counts.items() if count > 1]
        if duplicates:
            raise ValueError(f"column keys must be unique, found duplicates: {', '.join(duplicates)}")

        entries_uid = Env.get_uid()
        cols_uid = Env.get_uid()

        t = self
        t = t._localize_entries(entries_uid, cols_uid)

        def fmt(f, col_key):
            if f:
                return col_key + separator + f
            else:
                return col_key

        t = t.annotate(**{
            fmt(f, col_keys[i]): t[entries_uid][i][j]
            for i in range(len(col_keys))
            for j, f in enumerate(self.entry)
        })
        t = t.drop(cols_uid, entries_uid)

        return t

    @typecheck_method(rows=bool, cols=bool, entries=bool, handler=nullable(anyfunc))
    def summarize(self, *, rows=True, cols=True, entries=True, handler=None):
        """Compute and print summary information about the fields in the matrix table.

        .. include:: _templates/experimental.rst

        Parameters
        ----------
        rows : :obj:`bool`
            Compute summary for the row fields.
        cols : :obj:`bool`
            Compute summary for the column fields.
        entries : :obj:`bool`
            Compute summary for the entry fields.
        """

        if handler is None:
            handler = default_handler()
        if cols:
            handler(self.col._summarize(header='Columns', top=True))
        if rows:
            handler(self.row._summarize(header='Rows', top=True))
        if entries:
            handler(self.entry._summarize(header='Entries', top=True))

    def _write_block_matrix(self, path, overwrite, entry_field, block_size):
        mt = self
        mt = mt._select_all(entry_exprs={entry_field: mt[entry_field]})

        writer = ir.MatrixBlockMatrixWriter(path, overwrite, entry_field, block_size)
        Env.backend().execute(ir.MatrixWrite(self._mir, writer))

    def _calculate_new_partitions(self, n_partitions):
        """returns a set of range bounds that can be passed to write"""
        ht = self.rows()
        ht = ht.select().select_globals()
        return Env.backend().execute(ir.TableToValueApply(
            ht._tir,
            {'name': 'TableCalculateNewPartitions',
             'nPartitions': n_partitions}))


matrix_table_type.set(MatrixTable)
