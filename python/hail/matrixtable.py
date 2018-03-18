import itertools
from typing import *

import hail
import hail as hl
from hail.expr.expr_ast import Select, TopLevelReference
from hail.expr.expressions import *
from hail.expr.types import *
from hail.table import Table, ExprContainer
from hail.typecheck import *
from hail.utils import storage_level, LinkedList
from hail.utils.java import escape_id, warn, jiterable_to_list, Env, scala_object
from hail.utils.misc import get_nice_field_error, wrap_to_tuple, check_collisions, check_field_uniqueness


class GroupedMatrixTable(ExprContainer):
    """Matrix table grouped by row or column that can be aggregated to produce a new matrix table.

    There are only two operations on a grouped matrix table, :meth:`.GroupedMatrixTable.partition_hint`
    and :meth:`.GroupedMatrixTable.aggregate`.

    .. testsetup::

        dataset2 = dataset.annotate_globals(global_field=5)
        table1 = dataset.rows()
        table1 = table1.annotate_globals(global_field=5)
        table1 = table1.annotate(consequence='SYN')

        table2 = dataset.cols()
        table2 = table2.annotate(pop='AMR', is_case=False, sex='F')

    """

    def __init__(self, parent: 'MatrixTable', row_keys=None, col_keys=None):
        super(GroupedMatrixTable, self).__init__()
        self._parent = parent
        self._copy_fields_from(parent)
        self._row_keys = row_keys
        self._col_keys = col_keys
        self._partitions = None
        self._partition_key = None

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

    def _process_keys(self, left):
        row_keys = []
        col_keys = []
        renamed = {}
        new_row_fields = {}
        new_col_fields = {}

        if self._row_keys is not None:
            for k, v in self._row_keys.items():
                if v in self._parent._fields_inverse:
                    f = self._parent._fields_inverse[v]
                else:
                    f = Env.get_uid()
                    new_row_fields[f] = v
                row_keys.append(f)
                if k != f:
                    renamed[f] = k
            left = left.annotate_rows(**new_row_fields)
        if self._col_keys is not None:
            for k, v in self._col_keys.items():
                if v in self._parent._fields_inverse:
                    f = self._parent._fields_inverse[v]
                else:
                    f = Env.get_uid()
                new_col_fields[f] = v
                col_keys.append(f)
                if k != f:
                    renamed[f] = k
            left = left.annotate_cols(**new_col_fields)

        self._new_row_keys = row_keys
        self._new_col_keys = col_keys

        def cleanup(mt):
            return mt.rename(renamed)

        return left, cleanup

    def describe(self):
        """Print information about grouped matrix table."""

        if self._row_keys is None:
            rowstr = ""
        else:
            rowstr = "\nRows: \n" + "\n    ".join(["{}: {}".format(k, v._type) for k, v in self._row_keys.items()])
            if self._partition_key:
                rowstr += "\n  Partition by: {}".format(self._partition_key)

        if self._col_keys is None:
            colstr = ""
        else:
            colstr = "\nColumns: \n" + "\n    ".join(["{}: {}".format(k, v) for k, v in self._col_keys.items()])

        s = '----------------------------------------\n' \
            'GroupedMatrixTable grouped by {}\n' \
            '----------------------------------------\n' \
            'Parent MatrixTable:\n'.format(
            rowstr,
            colstr,
            self._partition_key)

        print(s)
        self._parent.describe()

    @typecheck_method(exprs=oneof(str, Expression),
                      named_exprs=expr_any)
    def group_rows_by(self,
                      *exprs: Tuple[Union[Expression, str]],
                      **named_exprs: NamedExprs) -> 'GroupedMatrixTable':
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
            ast = e._ast.expand()
            if any(not isinstance(a, TopLevelReference) and not isinstance(a, Select) for a in ast):
                raise ExpressionException("method 'group_rows_by' expects keyword arguments for complex expressions")
            key = ast[0].name

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
    def group_cols_by(self,
                      *exprs: FieldRefArgs,
                      **named_exprs: NamedExprs) -> 'GroupedMatrixTable':
        """Group rows, used with :meth:`.GroupedMatrixTable.aggregate`.

        Examples
        --------
        Aggregate to a matrix with cohort as column keys, computing the call rate
        as an entry field:

        .. testsetup::

            dataset = dataset.annotate_cols(cohort = 'cohort')

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
            ast = e._ast.expand()
            if any(not isinstance(a, TopLevelReference) and not isinstance(a, Select) for a in ast):
                raise ExpressionException("method 'group_cols_by' expects keyword arguments for complex expressions")
            key = ast[0].name
            if key in new_keys or key in kept_fields:
                raise ExpressionException("method 'group_cols_by' found duplicate field: {}".format(key))
            new_keys[key] = e

        for key, e in named_exprs.items():
            if key in new_keys or key in kept_fields:
                raise ExpressionException("method 'group_cols_by' found duplicate field: {}".format(key))
            new_keys[key] = e

        return GroupedMatrixTable(self._parent, col_keys=new_keys)

    def partition_by(self, *fields: Tuple[str]) -> 'GroupedMatrixTable':
        """Set the partition key.

        Parameters
        ----------
        fields : varargs of :obj:`str`
            Row partition key. Must be a prefix of the key. By default, the
            partition key is the entire key.

        Returns
        -------
        :class:`.GroupedMatrixTable`
            Self.
        """
        # FIXME: better docs
        if len(fields) == 0:
            raise ValueError('require at least one partition field')
        if not fields == tuple(name for name, _ in self._groups[:len(fields)]):
            raise ValueError('Expect partition fields to be a prefix of the keys {}'.format(
                ', '.join("'{}'".format(name) for name in self._keys)))
        self._partition_key = fields
        return self

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

    def aggregate(self, **named_exprs: NamedExprs) -> 'MatrixTable':
        """Aggregate by group, used after :meth:`.MatrixTable.group_rows_by` or :meth:`.MatrixTable.group_cols_by`.

        Examples
        --------
        Aggregate to a matrix with genes as row keys, computing the number of
        non-reference calls as an entry field:

        >>> dataset_result = (dataset.group_rows_by(dataset.gene)
        ...                          .aggregate(n_non_ref = agg.count_where(dataset.GT.is_non_ref())))

        Parameters
        ----------
        named_exprs : varargs of :class:`.Expression`
            Aggregation expressions.

        Returns
        -------
        :class:`.MatrixTable`
            Aggregated matrix table.
        """

        assert self._row_keys is not None or self._col_keys is not None

        named_exprs = {k: to_expr(v) for k, v in named_exprs.items()}

        strs = []

        fixed_fields = list(self._parent.globals.dtype)
        if self._row_keys is not None:
            fixed_fields.extend(self._row_keys.keys())
        else:
            fixed_fields.extend(list(self._parent.row.dtype))
        if self._col_keys is not None:
            fixed_fields.extend(self._col_keys.keys())
        else:
            fixed_fields.extend(list(self._parent.col.dtype))

        base, _ = self._parent._process_joins(*named_exprs.values())
        for k, v in named_exprs.items():
            if k in fixed_fields:
                raise ExpressionException("GroupedMatrixTable.aggregate cannot assign duplicate field '{}'".format(k))
            analyze('GroupedMatrixTable.aggregate', v, self._fixed_indices(),
                    {self._parent._row_axis, self._parent._col_axis})
            strs.append('{} = {}'.format(escape_id(k), v._ast.to_hql()))

        base, rename = self._process_keys(base)

        if self._col_keys is not None:
            assert self._new_col_keys is not None
            base = MatrixTable(base.key_cols_by(*self._new_col_keys)._jvds
                               .groupColsBy(','.join(["`{}` = sa.`{}`".format(k, k) for k in self._new_col_keys]),
                                            ',\n'.join(strs)))
        elif self._row_keys is not None:
            base = MatrixTable(
                base.key_rows_by(*self._new_row_keys, partition_key=self._partition_key)._jvds.aggregateRowsByKey(
                    ',\n'.join(strs)))
        else:
            raise ValueError("GroupedMatrixTable cannot be aggregated if no groupings are specified.")

        return rename(base)


matrix_table_type = lazy()


class MatrixTable(ExprContainer):
    """Hail's distributed implementation of a structured matrix.

    Use :func:`.read_matrix_table` to read a matrix table that was written with
    :meth:`.MatrixTable.write`.

    Examples
    --------

    .. testsetup::

        dataset2 = dataset.annotate_globals(global_field=5)
        table1 = dataset.rows()
        table1 = table1.annotate_globals(global_field=5)
        table1 = table1.annotate(consequence='SYN')

        table2 = dataset.cols()
        table2 = table2.annotate(pop='AMR', is_case=False, sex='F')

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

    def __init__(self, jvds):
        super(MatrixTable, self).__init__()
        self._jvds = jvds

        self._globals = None
        self._col_values = None

        self._row_axis = 'row'
        self._col_axis = 'column'

        self._global_indices = Indices(self, set())
        self._row_indices = Indices(self, {self._row_axis})
        self._col_indices = Indices(self, {self._col_axis})
        self._entry_indices = Indices(self, {self._row_axis, self._col_axis})

        self._global_type = HailType._from_java(jvds.globalType())
        self._col_type = HailType._from_java(jvds.colType())
        self._row_type = HailType._from_java(jvds.rowType())
        self._entry_type = HailType._from_java(jvds.entryType())

        assert isinstance(self._global_type, tstruct), self._global_type
        assert isinstance(self._col_type, tstruct), self._col_type
        assert isinstance(self._row_type, tstruct), self._row_type
        assert isinstance(self._entry_type, tstruct), self._entry_type

        self._globals = construct_expr(TopLevelReference('global', self._global_indices), self._global_type,
                                       indices=self._global_indices)
        self._row = construct_expr(TopLevelReference('va', self._row_indices), self._row_type,
                                   indices=self._row_indices)
        self._col = construct_expr(TopLevelReference('sa', self._col_indices), self._col_type,
                                   indices=self._col_indices)
        self._entry = construct_expr(TopLevelReference('g', self._entry_indices), self._entry_type,
                                     indices=self._entry_indices)

        self._partition_key = hail.struct(
            **{k: self._row[k] for k in jiterable_to_list(jvds.rowPartitionKey())})
        self._row_key = hail.struct(
            **{k: self._row[k] for k in jiterable_to_list(jvds.rowKey())})
        self._col_key = hail.struct(
            **{k: self._col[k] for k in jiterable_to_list(jvds.colKey())})

        self._num_samples = None

        for k, v in itertools.chain(self._globals.items(),
                                    self._row.items(),
                                    self._col.items(),
                                    self._entry.items()):
            self._set_field(k, v)

    @typecheck_method(item=oneof(str, sized_tupleof(oneof(slice, Expression, tupleof(Expression)),
                                                    oneof(slice, Expression, tupleof(Expression)))))
    def __getitem__(self, item):
        if isinstance(item, str):
            return self._get_field(item)
        else:
            # this is the join path
            exprs = item
            row_key = None
            if isinstance(exprs[0], slice):
                s = exprs[0]
                if not (s.start is None and s.stop is None and s.step is None):
                    raise ExpressionException(
                        "Expect unbounded slice syntax ':' to indicate axes of a MatrixTable, but found parameter(s) [{}]".format(
                            ', '.join(x for x in ['start' if s.start is not None else None,
                                                  'stop' if s.stop is not None else None,
                                                  'step' if s.step is not None else None] if x is not None)
                        )
                    )
            else:
                row_key = [to_expr(e) for e in wrap_to_tuple(exprs[0])]
                key_types = [k.dtype for k in row_key]
                expected = [self[k].dtype for k in self.row_key]
                if key_types != expected:
                    raise ExpressionException(
                        'Type mismatch for MatrixTable row key: expected [{}], found [{}]'.format(
                            ', '.join(map(str, expected)), ', '.join(map(str, key_types))))

            col_key = None
            if isinstance(exprs[1], slice):
                s = exprs[1]
                if not (s.start is None and s.stop is None and s.step is None):
                    raise ExpressionException(
                        "Expect unbounded slice syntax ':' to indicate axes of a MatrixTable, but found parameter(s) [{}]".format(
                            ', '.join(x for x in ['start' if s.start is not None else None,
                                                  'stop' if s.stop is not None else None,
                                                  'step' if s.step is not None else None] if x is not None)
                        )
                    )
            else:
                col_key = [to_expr(e) for e in wrap_to_tuple(exprs[1])]
                key_types = [k.dtype for k in col_key]
                expected = [self[k].dtype for k in self.col_key]
                if key_types != expected:
                    raise ExpressionException(
                        'Type mismatch for MatrixTable column key: expected [{}], found [{}]'.format(
                            ', '.join(map(str, expected)), ', '.join(map(str, key_types))))

            if row_key is not None and col_key is not None:
                return self.index_entries(row_key, col_key)
            elif row_key is not None and col_key is None:
                return self.index_rows(*row_key)
            elif row_key is None and col_key is not None:
                return self.index_cols(*col_key)
            else:
                return self.index_globals()

    @property
    def col_key(self) -> StructExpression:
        """Column key struct.

        Examples
        --------

        Get the column key field names:

        .. doctest::

            >>> list(dataset.col_key)
            ['s']

        Returns
        -------
        :class:`.StructExpression`
        """
        return self._col_key

    @property
    def row_key(self) -> StructExpression:
        """Row key struct.

        Examples
        --------

        Get the row key field names:

        .. doctest::

            >>> list(dataset.row_key)
            ['locus', 'alleles']

        Returns
        -------
        :class:`.StructExpression`
        """
        return self._row_key

    @property
    def partition_key(self) -> StructExpression:
        """Partition key struct.

        Examples
        --------

        Get the partition key field names:

        .. doctest::

            >>> list(dataset.partition_key)
            ['locus']

        Returns
        -------
        :class:`.StructExpression`
        """
        return self._partition_key

    @property
    def globals(self) -> StructExpression:
        """Returns a struct expression including all global fields.

        Returns
        -------
        :class:`.StructExpression`
        """
        return self._globals

    @property
    def row(self) -> StructExpression:
        """Returns a struct expression including all row-indexed fields.

        Examples
        --------

        Get the first five row field names:

        .. doctest::

            >>> list(dataset.row)[:5]
            ['locus', 'alleles', 'rsid', 'qual', 'filters']

        Returns
        -------
        :class:`.StructExpression`
            Struct of all row fields.
        """
        return self._row

    @property
    def col(self) -> StructExpression:
        """Returns a struct expression including all column-indexed fields.

        Examples
        --------

        Get all column field names:

        .. doctest::

            >>> list(dataset.col)
            ['s', 'sample_qc', 'is_case', 'pheno', 'cov', 'cov1', 'cov2', 'cohorts', 'pop']

        Returns
        -------
        :class:`.StructExpression`
            Struct of all column fields.
        """
        return self._col

    @property
    def entry(self) -> StructExpression:
        """Returns a struct expression including all row-and-column-indexed fields.

        Examples
        --------

        Get all entry field names:

        .. doctest::

            >>> list(dataset.entry)
            ['GT', 'AD', 'DP', 'GQ', 'PL']


        Returns
        -------
        :class:`.StructExpression`
            Struct of all entry fields.
        """
        return self._entry

    @typecheck_method(keys=oneof(str, Expression))
    def key_cols_by(self, *keys: FieldRefArgs) -> 'MatrixTable':
        """Key columns by a new set of fields.

        Parameters
        ----------
        keys : varargs of :obj:`str`
            Column fields to key by.
        Returns
        -------
        :class:`.MatrixTable`
        """
        str_keys = []
        for k in keys:
            if isinstance(k, Expression):
                if k not in self._fields_inverse:
                    raise ExpressionException("'key_cols_by' permits only top-level fields of the matrix table")
                elif k._indices != self._col_indices:
                    raise ExpressionException("key_cols_by' expects column fields, found index {}"
                                              .format(list(k._indices.axes)))
                str_keys.append(self._fields_inverse[k])
            else:
                if k not in self._fields:
                    raise LookupError(get_nice_field_error(self, k))
                if not self._fields[k]._indices == self._col_indices:
                    raise ValueError("'{}' is not a column field".format(k))
                str_keys.append(k)
        return MatrixTable(self._jvds.keyColsBy(str_keys))

    @typecheck_method(keys=oneof(str, Expression),
                      partition_key=nullable(oneof(oneof(str, Expression), listof(oneof(str, Expression)))))
    def key_rows_by(self,
                    *keys: FieldRefArgs,
                    partition_key: Optional[Union[FieldRef, Sequence[FieldRef]]] = None
                    ) -> 'MatrixTable':
        """Key rows by a new set of fields.

        Parameters
        ----------
        keys : varargs of :obj:`str` or :class:`.Expression`.
            Row fields to key by.
        partition_key : :obj:`str` or :class:`.Expression`, or :obj:`list` of :obj:`str` or :class:`.Expression`, optional
            Row fields to partition by. Must be a prefix of the key.
            Default: all keys.
        Returns
        -------
        :class:`.MatrixTable`
        """

        def check_ref(key):
            if isinstance(k, Expression):
                if k not in self._fields_inverse:
                    raise ExpressionException("'key_rows_by' permits only top-level fields of the matrix table")
                elif k._indices != self._row_indices:
                    raise ExpressionException("key_rows_by' expects row fields, found index {}"
                                              .format(list(k._indices.axes)))
                return self._fields_inverse[k]
            else:
                if not self._get_field(k)._indices == self._row_indices:
                    raise ValueError("'{}' is not a row field".format(k))
                return key

        str_keys = []

        for k in keys:
            str_keys.append(check_ref(k))

        if partition_key == None:
            str_pks = str_keys
        else:
            str_pks = []
            for k in wrap_to_tuple(partition_key):
                str_pks.append(check_ref(k))

        return MatrixTable(self._jvds.keyRowsBy(str_keys, str_pks))

    def annotate_globals(self, **named_exprs: NamedExprs) -> 'MatrixTable':
        """Create new global fields by name.

        Examples
        --------
        Add two global fields:

        >>> pops_1kg = {'EUR', 'AFR', 'EAS', 'SAS', 'AMR'}
        >>> dataset_result = dataset.annotate_globals(pops_in_1kg = pops_1kg,
        ...                                           gene_list = ['SHH', 'SCN1A', 'SPTA1', 'DISC1'])

        Add global fields from another table and matrix table:

        >>> dataset_result = dataset.annotate_globals(thing1 = dataset2[:, :].global_field,
        ...                                           thing2 = table1[:].global_field)

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
        exprs = []
        named_exprs = {k: to_expr(v) for k, v in named_exprs.items()}
        base, cleanup = self._process_joins(*named_exprs.values())

        for k, v in named_exprs.items():
            analyze('MatrixTable.annotate_globals', v, self._global_indices)
            exprs.append('global.{k} = {v}'.format(k=escape_id(k), v=v._ast.to_hql()))
            check_collisions(self._fields, k, self._global_indices)
        m = MatrixTable(base._jvds.annotateGlobalExpr(",\n".join(exprs)))
        return cleanup(m)

    def annotate_rows(self, **named_exprs: NamedExprs) -> 'MatrixTable':
        """Create new row-indexed fields by name.

        Examples
        --------
        Compute call statistics for high quality samples per variant:

        >>> high_quality_calls = agg.filter(dataset.sample_qc.gq_mean > 20, dataset.GT)
        >>> dataset_result = dataset.annotate_rows(call_stats = agg.call_stats(high_quality_calls, dataset.alleles))

        Add functional annotations from a :class:`.Table` keyed by :class:`.TVariant`:, and another
        :class:`.MatrixTable`.

        >>> dataset_result = dataset.annotate_rows(consequence = table1[dataset.locus, dataset.alleles].consequence,
        ...                                        dataset2_AF = dataset2[(dataset.locus, dataset.alleles), :].info.AF)

        Note
        ----
        This method supports aggregation over columns. For instance, the usage:

        >>> dataset_result = dataset.annotate_rows(mean_GQ = agg.mean(dataset.GQ))

        will compute the mean per row.

        Notes
        -----
        This method creates new row fields, but can also overwrite existing fields. Only
        same-scope fields can be overwritten: for example, it is not possible to annotate a
        global field `foo` and later create an row field `foo`. However, it would be possible
        to create an row field `foo` and later create another row field `foo`, overwriting
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
            Matrix table with new row-indexed field(s).
        """
        exprs = []
        named_exprs = {k: to_expr(v) for k, v in named_exprs.items()}
        base, cleanup = self._process_joins(*named_exprs.values())

        for k, v in named_exprs.items():
            analyze('MatrixTable.annotate_rows', v, self._row_indices, {self._col_axis})
            exprs.append('{k} = {v}'.format(k=escape_id(k), v=v._ast.to_hql()))
            check_collisions(self._fields, k, self._row_indices)
        m = MatrixTable(base._jvds.annotateRowsExpr(",\n".join(exprs)))
        return cleanup(m)

    def annotate_cols(self, **named_exprs: NamedExprs) -> 'MatrixTable':
        """Create new column-indexed fields by name.

        Examples
        --------
        Compute statistics about the GQ distribution per sample:

        >>> dataset_result = dataset.annotate_cols(sample_gq_stats = agg.stats(dataset.GQ))

        Add sample metadata from a :class:`.hail.Table`.

        >>> dataset_result = dataset.annotate_cols(population = table2[dataset.s].pop)

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
        exprs = []
        named_exprs = {k: to_expr(v) for k, v in named_exprs.items()}
        base, cleanup = self._process_joins(*named_exprs.values())

        for k, v in named_exprs.items():
            analyze('MatrixTable.annotate_cols', v, self._col_indices, {self._row_axis})
            exprs.append('{k} = {v}'.format(k=escape_id(k), v=v._ast.to_hql()))
            check_collisions(self._fields, k, self._col_indices)
        m = MatrixTable(base._jvds.annotateColsExpr(",\n".join(exprs)))
        return cleanup(m)

    def annotate_entries(self, **named_exprs: NamedExprs) -> 'MatrixTable':
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
        exprs = []
        named_exprs = {k: to_expr(v) for k, v in named_exprs.items()}
        base, cleanup = self._process_joins(*named_exprs.values())

        for k, v in named_exprs.items():
            analyze('MatrixTable.annotate_entries', v, self._entry_indices)
            exprs.append('g.{k} = {v}'.format(k=escape_id(k), v=v._ast.to_hql()))
            check_collisions(self._fields, k, self._entry_indices)
        m = MatrixTable(base._jvds.annotateEntriesExpr(",\n".join(exprs)))
        return cleanup(m)

    def select_globals(self, *exprs: FieldRefArgs, **named_exprs: NamedExprs) -> 'MatrixTable':
        """Select existing global fields or create new fields by name, dropping the rest.

        Examples
        --------
        Select one existing field and compute a new one:

        .. testsetup::

            dataset = dataset.annotate_globals(global_field_1 = 5, global_field_2 = 10)

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
        exprs = [to_expr(e) if not isinstance(e, str) else self[e] for e in exprs]
        named_exprs = {k: to_expr(v) for k, v in named_exprs.items()}
        strs = []
        all_exprs = []
        base, cleanup = self._process_joins(*itertools.chain(exprs, named_exprs.values()))

        ids = []
        for e in exprs:
            all_exprs.append(e)
            analyze('MatrixTable.select_globals', e, self._global_indices)
            if e._ast.search(lambda ast: not isinstance(ast, TopLevelReference) and not isinstance(ast, Select)):
                raise ExpressionException("method 'select_globals' expects keyword arguments for complex expressions")
            strs.append('{}: {}'.format(escape_id(e._ast.name), e._ast.to_hql()))
            ids.append(e._ast.name)

        for k, e in named_exprs.items():
            all_exprs.append(e)
            analyze('MatrixTable.select_globals', e, self._global_indices)
            check_collisions(self._fields, k, self._global_indices)
            strs.append('{}: {}'.format(escape_id(k), to_expr(e)._ast.to_hql()))
            ids.append(k)
        check_field_uniqueness(ids)
        m = MatrixTable(base._jvds.annotateGlobalExpr('global = {' + ',\n'.join(strs) + '}'))
        return cleanup(m)

    def select_rows(self, *exprs: FieldRefArgs, **named_exprs: NamedExprs) -> 'MatrixTable':
        """Select existing row fields or create new fields by name, dropping the rest.

        Examples
        --------
        Select existing fields and compute a new one:

        >>> dataset_result = dataset.select_rows(
        ...    dataset.locus,
        ...    dataset.alleles,
        ...    dataset.variant_qc.gq_mean,
        ...    high_quality_cases = agg.count_where((dataset.GQ > 20) &
        ...                                         dataset.is_case))

        Notes
        -----
        This method creates new row fields. If a created field shares its name
        with a differently-indexed field of the table, the method will fail.

        Note
        ----

        See :meth:`.Table.select` for more information about using ``select`` methods.

        Note
        ----
        This method supports aggregation over columns. For instance, the usage:

        >>> dataset_result = dataset.select_rows(dataset.locus, dataset.alleles, mean_GQ = agg.mean(dataset.GQ))

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
        exprs = [to_expr(e) if not isinstance(e, str) else self[e] for e in exprs]
        named_exprs = {k: to_expr(v) for k, v in named_exprs.items()}
        strs = []
        all_exprs = []
        base, cleanup = self._process_joins(*itertools.chain(exprs, named_exprs.values()))

        ids = []
        for e in exprs:
            all_exprs.append(e)
            analyze('MatrixTable.select_rows', e, self._row_indices, {self._col_axis})
            if e._ast.search(lambda ast: not isinstance(ast, TopLevelReference) and not isinstance(ast, Select)):
                raise ExpressionException("method 'select_rows' expects keyword arguments for complex expressions")
            strs.append(e._ast.to_hql())
            ids.append(e._ast.name)
        for k, e in named_exprs.items():
            all_exprs.append(e)
            analyze('MatrixTable.select_rows', e, self._row_indices, {self._col_axis})
            check_collisions(self._fields, k, self._row_indices)
            strs.append('{} = {}'.format(escape_id(k), e._ast.to_hql()))
            ids.append(k)
        check_field_uniqueness(ids)
        m = MatrixTable(base._jvds.selectRows(strs))
        return cleanup(m)

    def select_cols(self, *exprs: FieldRefArgs, **named_exprs: NamedExprs) -> 'MatrixTable':
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

        exprs = [to_expr(e) if not isinstance(e, str) else self[e] for e in exprs]
        named_exprs = {k: to_expr(v) for k, v in named_exprs.items()}
        strs = []
        all_exprs = []
        base, cleanup = self._process_joins(*itertools.chain(exprs, named_exprs.values()))

        ids = []
        for e in exprs:
            all_exprs.append(e)
            analyze('MatrixTable.select_cols', e, self._col_indices, {self._row_axis})
            if e._ast.search(lambda ast: not isinstance(ast, TopLevelReference) and not isinstance(ast, Select)):
                raise ExpressionException("method 'select_cols' expects keyword arguments for complex expressions")
            strs.append(e._ast.to_hql())
            ids.append(e._ast.name)
        for k, e in named_exprs.items():
            all_exprs.append(e)
            analyze('MatrixTable.select_cols', e, self._col_indices, {self._row_axis})
            check_collisions(self._fields, k, self._col_indices)
            strs.append('{} = {}'.format(escape_id(k), e._ast.to_hql()))
            ids.append(k)
        check_field_uniqueness(ids)
        m = MatrixTable(base._jvds.selectCols(strs))
        return cleanup(m)

    def select_entries(self, *exprs: FieldRefArgs, **named_exprs: NamedExprs) -> 'MatrixTable':
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
        exprs = [to_expr(e) if not isinstance(e, str) else self[e] for e in exprs]
        named_exprs = {k: to_expr(v) for k, v in named_exprs.items()}
        strs = []
        all_exprs = []
        base, cleanup = self._process_joins(*itertools.chain(exprs, named_exprs.values()))

        ids = []
        for e in exprs:
            all_exprs.append(e)
            analyze('MatrixTable.select_entries', e, self._entry_indices)
            if not e._indices == self._entry_indices:
                # detect row or col fields here
                raise ExpressionException("method 'select_entries' parameter 'exprs' expects entry-indexed fields,"
                                          " found indices {}".format(list(e._indices.axes)))
            if e._ast.search(lambda ast: not isinstance(ast, TopLevelReference) and not isinstance(ast, Select)):
                raise ExpressionException("method 'select_entries' expects keyword arguments for complex expressions")
            strs.append(e._ast.to_hql())
            ids.append(e._ast.name)
        for k, e in named_exprs.items():
            all_exprs.append(e)
            analyze('MatrixTable.select_entries', e, self._entry_indices)
            check_collisions(self._fields, k, self._entry_indices)
            strs.append('{} = {}'.format(escape_id(k), e._ast.to_hql()))
            ids.append(k)
        check_field_uniqueness(ids)
        m = MatrixTable(base._jvds.selectEntries(strs))
        return cleanup(m)

    @typecheck_method(exprs=oneof(str, Expression))
    def drop(self, *exprs: FieldRefArgs) -> 'MatrixTable':
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
        if any(self._fields[field]._indices == self._global_indices for field in fields_to_drop):
            # need to drop globals
            new_global_fields = [f for f in m.globals if f not in fields_to_drop]
            m = m.select_globals(*new_global_fields)

        row_fields = [x for x in fields_to_drop if self._fields[x]._indices == self._row_indices]
        if row_fields:
            # need to drop row fields
            m = MatrixTable(m._jvds.dropRows(row_fields))

        if any(self._fields[field]._indices == self._col_indices for field in fields_to_drop):
            # need to drop col fields
            new_col_fields = [f for f in m.col if f not in fields_to_drop]
            m = m.select_cols(*new_col_fields)

        entry_fields = [x for x in fields_to_drop if self._fields[x]._indices == self._entry_indices]
        if any(self._fields[field]._indices == self._entry_indices for field in fields_to_drop):
            # need to drop entry fields
            m = MatrixTable(m._jvds.dropEntries(entry_fields))

        return m

    def drop_rows(self):
        """Drop all rows of the matrix.  Is equivalent to:

        >>> dataset_result = dataset.filter_rows(False)

        .. include:: _templates/experimental.rst

        Returns
        -------
        :class:`.MatrixTable`
            Matrix table with no rows.
        """
        warn("deprecation: 'drop_rows' will be removed before 0.2 release")
        return MatrixTable(self._jvds.dropRows())

    def drop_cols(self):
        """Drop all columns of the matrix.  Is equivalent to:

        >>> dataset_result = dataset.filter_cols(False)

        .. include:: _templates/experimental.rst

        Returns
        -------
        :class:`.MatrixTable`
            Matrix table with no columns.
        """
        warn("deprecation: 'drop_cols' will be removed before 0.2 release")
        return MatrixTable(self._jvds.dropCols())

    @typecheck_method(expr=expr_any, keep=bool)
    def filter_rows(self, expr: BooleanExpression, keep: bool = True) -> 'MatrixTable':
        """Filter rows of the matrix.

        Examples
        --------

        Keep rows where `variant_qc.AF` is below 1%:

        >>> dataset_result = dataset.filter_rows(dataset.variant_qc.AF < 0.01, keep=True)

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
        base, cleanup = self._process_joins(expr)
        analyze('MatrixTable.filter_rows', expr, self._row_indices, {self._col_axis})
        m = MatrixTable(base._jvds.filterRowsExpr(expr._ast.to_hql(), keep))
        return cleanup(m)

    @typecheck_method(expr=expr_bool, keep=bool)
    def filter_cols(self, expr: BooleanExpression, keep: bool = True) -> 'MatrixTable':
        """Filter columns of the matrix.

        Examples
        --------

        Keep columns where `pheno.is_case` is ``True`` and `pheno.age` is larger
        than 50:

        >>> dataset_result = dataset.filter_cols(dataset.pheno.is_case &
        ...                                      (dataset.pheno.age > 50),
        ...                                      keep=True)

        Remove rows where `sample_qc.gq_mean` is less than 20:

        >>> dataset_result = dataset.filter_cols(dataset.sample_qc.gq_mean < 20,
        ...                                      keep=False)

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
        expr = to_expr(expr)
        base, cleanup = self._process_joins(expr)
        analyze('MatrixTable.filter_cols', expr, self._col_indices, {self._row_axis})

        m = MatrixTable(base._jvds.filterColsExpr(expr._ast.to_hql(), keep))
        return cleanup(m)

    @typecheck_method(expr=expr_bool, keep=bool)
    def filter_entries(self, expr: BooleanExpression, keep: bool = True) -> 'MatrixTable':
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
        expr = to_expr(expr)
        base, cleanup = self._process_joins(expr)
        analyze('MatrixTable.filter_entries', expr, self._entry_indices)

        m = MatrixTable(base._jvds.filterEntries(expr._ast.to_hql(), keep))
        return cleanup(m)

    def transmute_globals(self, **named_exprs: NamedExprs) -> 'MatrixTable':
        """Similar to :meth:`.MatrixTable.annotate_globals`, but drops referenced fields.

        Note
        ----
        Not implemented.

        Parameters
        ----------
        named_exprs : keyword args of :class:`.Expression`
            Annotation expressions.

        Returns
        -------
        :class:`.MatrixTable`
            Annotated matrix table.
        """
        raise NotImplementedError()

    def transmute_rows(self, **named_exprs: NamedExprs) -> 'MatrixTable':
        """Similar to :meth:`.MatrixTable.annotate_rows`, but drops referenced fields.

        Note
        ----
        Not implemented.

        Parameters
        ----------
        named_exprs : keyword args of :class:`.Expression`
            Annotation expressions.

        Returns
        -------
        :class:`.MatrixTable`
            Annotated matrix table.
        """

        raise NotImplementedError()

    def transmute_cols(self, **named_exprs: NamedExprs) -> 'MatrixTable':
        """Similar to :meth:`.MatrixTable.annotate_cols`, but drops referenced fields.

        Note
        ----
        Not implemented.

        Parameters
        ----------
        named_exprs : keyword args of :class:`.Expression`
            Annotation expressions.

        Returns
        -------
        :class:`.MatrixTable`
            Annotated matrix table.
        """
        raise NotImplementedError()

    def transmute_entries(self, **named_exprs):
        """Similar to :meth:`.MatrixTable.annotate_entries`, but drops referenced fields.

        Note
        ----
        Not implemented.

        Parameters
        ----------
        named_exprs : keyword args of :class:`.Expression`
            Annotation expressions.

        Returns
        -------
        :class:`.MatrixTable`
            Annotated matrix table.
        """
        raise NotImplementedError()

    @typecheck_method(expr=expr_any)
    def aggregate_rows(self, expr: Expression) -> Any:
        """Aggregate over rows to a local value.

        Examples
        --------
        Aggregate over rows:

        .. doctest::

            >>> dataset.aggregate_rows(hl.struct(n_high_quality=agg.count_where(dataset.qual > 40),
            ...                                  mean_qual=agg.mean(dataset.qual)))
            Struct(n_high_quality=100150224, mean_qual=50.12515572)

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

        result_json = base._jvds.aggregateRowsJSON(expr._ast.to_hql())
        return expr.dtype._from_json(result_json)

    @typecheck_method(expr=expr_any)
    def aggregate_cols(self, expr: Expression) -> Any:
        """Aggregate over columns to a local value.

        Examples
        --------
        Aggregate over columns:

        .. doctest::

            >>> dataset.aggregate_cols(
            ...    hl.struct(fraction_female=agg.fraction(dataset.pheno.is_female),
            ...              case_ratio=agg.count_where(dataset.is_case) / agg.count()))
            Struct(fraction_female=0.5102222, case_ratio=0.35156)

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

        result_json = base._jvds.aggregateColsJSON(expr._ast.to_hql())
        return expr.dtype._from_json(result_json)

    @typecheck_method(expr=expr_any)
    def aggregate_entries(self, expr: Expression) -> Any:
        """Aggregate over entries to a local value.

        Examples
        --------
        Aggregate over entries:

        .. doctest::

            >>> dataset.aggregate_entries(hl.struct(global_gq_mean=agg.mean(dataset.GQ),
            ...                                     call_rate=agg.fraction(hl.is_defined(dataset.GT))))
            Struct(global_gq_mean=31.16200, call_rate=0.981682)

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

        result_json = base._jvds.aggregateEntriesJSON(expr._ast.to_hql())
        return expr.dtype._from_json(result_json)

    @typecheck_method(field_expr=oneof(str, Expression))
    def explode_rows(self, field_expr: FieldRef) -> 'MatrixTable':
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
            s = 'va.{}'.format(escape_id(field_expr))
        else:
            analyze('MatrixTable.explode_rows', field_expr, self._row_indices, set(self._fields.keys()))
            if field_expr._ast.search(
                    lambda ast: not isinstance(ast, TopLevelReference) and not isinstance(ast, Select)):
                raise ExpressionException(
                    "method 'explode_rows' requires a field or subfield, not a complex expression")
            s = field_expr._ast.to_hql()
        return MatrixTable(self._jvds.explodeRows(s))

    @typecheck_method(field_expr=oneof(str, Expression))
    def explode_cols(self, field_expr: FieldRef) -> 'MatrixTable':
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
        :class:MatrixTable`
            Matrix table exploded column-wise for each element of `field_expr`.
        """

        if isinstance(field_expr, str):
            if not field_expr in self._fields:
                raise KeyError("MatrixTable has no field '{}'".format(field_expr))
            elif self._fields[field_expr]._indices != self._col_indices:
                raise ExpressionException("Method 'explode_cols' expects a field indexed by col, found axes '{}'"
                                          .format(self._fields[field_expr]._indices.axes))
            s = 'sa.{}'.format(escape_id(field_expr))
        else:
            analyze('MatrixTable.explode_cols', field_expr, self._col_indices)
            if field_expr._ast.search(
                    lambda ast: not isinstance(ast, TopLevelReference) and not isinstance(ast, Select)):
                raise ExpressionException(
                    "method 'explode_cols' requires a field or subfield, not a complex expression")
            s = field_expr._ast.to_hql()
        return MatrixTable(self._jvds.explodeCols(s))

    @typecheck_method(exprs=oneof(str, Expression), named_exprs=expr_any)
    def group_rows_by(self, *exprs: FieldRefArgs, **named_exprs: NamedExprs) -> 'GroupedMatrixTable':
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
    def group_cols_by(self, *exprs: FieldRefArgs, **named_exprs: NamedExprs) -> 'GroupedMatrixTable':
        """Group rows, used with :meth:`.GroupedMatrixTable.aggregate`.

        Examples
        --------
        Aggregate to a matrix with cohort as column keys, computing the call rate
        as an entry field:

        .. testsetup::

            dataset = dataset.annotate_cols(cohort = 'cohort')

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
            ast = e._ast.expand()
            if any(not isinstance(a, TopLevelReference) and not isinstance(a, Select) for a in ast):
                raise ExpressionException("method 'group_cols_by' expects keyword arguments for complex expressions")
            key = ast[0].name
            if key in new_keys:
                raise ExpressionException("method 'group_cols_by' found duplicate field: {}".format(key))
            new_keys.append(key)

        ds = self.annotate_cols(**named_exprs)
        for key in named_exprs.keys():
            if key in new_keys:
                raise ExpressionException("method 'group_cols_by' found duplicate field: {}".format(key))
            new_keys.append(key)

        return GroupedMatrixTable(self).group_cols_by(*exprs, **named_exprs)

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
        return self._jvds.countRows()

    def _force_count_rows(self):
        return self._jvds.forceCountRows()

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
        return self._jvds.numCols()

    def count(self) -> Tuple[int, int]:
        """Count the number of rows and columns in the matrix.

        Examples
        --------
        .. doctest::

            >>> dataset.count()

        Returns
        -------
        :obj:`int`, :obj:`int`
            Number of rows, number of cols.
        """
        r = self._jvds.count()
        return r._1(), r._2()

    @typecheck_method(output=str,
                      overwrite=bool,
                      _codec_spec=nullable(str))
    def write(self, output: str, overwrite: bool = False, _codec_spec: Optional[str] = None):
        """Write to disk.

        Examples
        --------

        >>> dataset.write('output/dataset.vds')

        Note
        ----
        The write path must end in ".vds".

        Warning
        -------
        Do not write to a path that is being read from in the same computation.

        Parameters
        ----------
        output : str
            Path at which to write.
        overwrite : bool
            If ``True``, overwrite an existing file at the destination.
        """

        self._jvds.write(output, overwrite, _codec_spec)

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
        return Table(self._jvds.globalsTable())

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
        return Table(self._jvds.rowsTable())

    def cols(self) -> Table:
        """Returns a table with all column fields in the matrix.

        Examples
        --------
        Extract the column table:

        >>> cols_table = dataset.cols()

        Returns
        -------
        :class:`.Table`
            Table with all column fields from the matrix, with one row per column of the matrix.
        """
        return Table(self._jvds.colsTable())

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

        Returns
        -------
        :class:`.Table`
            Table with all non-global fields from the matrix, with **one row per entry of the matrix**.
        """
        return Table(self._jvds.entriesTable())

    def index_globals(self) -> Expression:
        uid = Env.get_uid()

        def joiner(obj):
            if isinstance(obj, MatrixTable):
                return MatrixTable(Env.jutils().joinGlobals(obj._jvds, self._jvds, uid))
            else:
                assert isinstance(obj, Table)
                return Table(Env.jutils().joinGlobals(obj._jt, self._jvds, uid))

        return construct_expr(Select(TopLevelReference('global', Indices()), uid), self.globals.dtype,
                              joins=LinkedList(Join).push(Join(joiner, [uid], uid, [])))

    def index_rows(self, *exprs: Tuple[Expression]) -> StructExpression:
        exprs = [to_expr(e) for e in exprs]
        indices, aggregations, joins = unify_all(*exprs)
        src = indices.source

        if aggregations:
            raise ExpressionException('Cannot join using an aggregated field')
        uid = Env.get_uid()
        uids_to_delete = [uid]

        if src is None:
            raise ExpressionException('Cannot index with a scalar expression')

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
            is_partition_key = len(exprs) == len(src.partition_key) and all(
                exprs[i] is src.partition_key[i] for i in range(len(exprs)))

            if is_row_key or is_partition_key:
                prefix = 'va'
                joiner = lambda left: (
                    MatrixTable(left._jvds.annotateRowsVDS(right._jvds, uid)))
                schema = tstruct(**{f: t for f, t in self.row.dtype.items() if f not in self.row_key})
                return construct_expr(Select(TopLevelReference(prefix, src._row_indices), uid),
                                      schema, indices, aggregations,
                                      joins.push(Join(joiner, uids_to_delete, uid, exprs)))
            else:
                return self.rows().index(*exprs)

    def index_cols(self, *exprs: Tuple[Expression]) -> StructExpression:
        exprs = [to_expr(e) for e in exprs]
        indices, aggregations, joins = unify_all(*exprs)
        src = indices.source

        if aggregations:
            raise ExpressionException('Cannot join using an aggregated field')
        uid = Env.get_uid()

        if src is None:
            raise ExpressionException('Cannot index with a scalar expression')

        return self.cols().index(*exprs)

    def index_entries(self, row_exprs: Tuple[Expression], col_exprs: Tuple[Expression]) -> StructExpression:
        row_exprs = [to_expr(e) for e in row_exprs]
        col_exprs = [to_expr(e) for e in col_exprs]

        indices, aggregations, joins = unify_all(*(row_exprs + col_exprs))
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
                localized = Table(self._jvds.localizeEntries(row_uid))
                src_cols_indexed = self.cols().add_index(col_uid)
                src_cols_indexed = src_cols_indexed.annotate(**{col_uid: hl.int32(src_cols_indexed[col_uid])})
                left = left._annotate_all(row_exprs = {row_uid: localized.index(*row_exprs)[row_uid]},
                                          col_exprs = {col_uid: src_cols_indexed.index(*col_exprs)[col_uid]})

                return left.annotate_entries(**{uid: left[row_uid][left[col_uid]]})

            return construct_expr(Select(TopLevelReference('g', self._entry_indices), uid),
                                  self.entry.dtype, indices, aggregations,
                                  joins.push(Join(joiner, uids, uid, [*row_exprs, *col_exprs])))

    @typecheck_method(row_exprs=dictof(str, expr_any),
                      col_exprs=dictof(str, expr_any),
                      entry_exprs=dictof(str, expr_any),
                      global_exprs=dictof(str, expr_any))
    def _annotate_all(self,
                      row_exprs: Optional[Dict[str, Expression]] = {},
                      col_exprs: Optional[Dict[str, Expression]] = {},
                      entry_exprs: Optional[Dict[str, Expression]] = {},
                      global_exprs: Optional[Dict[str, Expression]] = {},
                      ) -> 'MatrixTable':
        all_exprs = list(itertools.chain(row_exprs.values(),
                                         col_exprs.values(),
                                         entry_exprs.values(),
                                         global_exprs.values()))

        base, cleanup = self._process_joins(*all_exprs)
        jmt = base._jvds
        if row_exprs:
            row_strs = []
            for k, v in row_exprs.items():
                analyze('MatrixTable.annotate_rows', v, self._row_indices, {self._col_axis})
                row_strs.append('{k} = {v}'.format(k=escape_id(k), v=v._ast.to_hql()))
                check_collisions(self._fields, k, self._row_indices)
                jmt = jmt.annotateRowsExpr(",\n".join(row_strs))
        if col_exprs:
            col_strs = []
            for k, v in col_exprs.items():
                analyze('MatrixTable.annotate_cols', v, self._col_indices, {self._row_axis})
                col_strs.append('{k} = {v}'.format(k=escape_id(k), v=v._ast.to_hql()))
                check_collisions(self._fields, k, self._col_indices)
                jmt = jmt.annotateColsExpr(",\n".join(col_strs))
        if entry_exprs:
            entry_strs = []
            for k, v in entry_exprs.items():
                analyze('MatrixTable.annotate_entries', v, self._entry_indices)
                entry_strs.append('g.{k} = {v}'.format(k=escape_id(k), v=v._ast.to_hql()))
                check_collisions(self._fields, k, self._entry_indices)
                jmt = jmt.annotateEntriesExpr(",\n".join(entry_strs))
        if global_exprs:
            global_strs = []
            for k, v in global_exprs.items():
                analyze('MatrixTable.annotate_globals', v, self._global_indices)
                global_strs.append('global.{k} = {v}'.format(k=escape_id(k), v=v._ast.to_hql()))
                check_collisions(self._fields, k, self._global_indices)
                jmt = jmt.annotateGlobalsExpr(",\n".join(global_strs))

        return cleanup(MatrixTable(jmt))

    def _process_joins(self, *exprs):

        all_uids = []
        left = self
        used_uids = set()

        for e in exprs:
            for j in list(e._joins)[::-1]:
                if j.uid not in used_uids:
                    left = j.join_function(left)
                    all_uids.extend(j.temp_vars)
                    used_uids.add(j.uid)

        def cleanup(matrix):
            remaining_uids = [uid for uid in all_uids if uid in matrix._fields]
            return matrix.drop(*remaining_uids)

        return left, cleanup

    def describe(self):
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

        row_key = ''.join("\n    '{name}': {type} ".format(name=f, type=format_type(self[f].dtype))
                          for f in self.row_key) if self.row_key else '\n    None'
        partition_key = ''.join("\n    '{name}': {type} ".format(name=f, type=format_type(self[f].dtype))
                                for f in self.partition_key) if self.partition_key else '\n    None'

        if len(self.col) == 0:
            col_fields = '\n    None'
        else:
            col_fields = ''.join("\n    '{name}': {type} ".format(
                name=f, type=format_type(t)) for f, t in self.col.dtype.items())

        col_key = ''.join("\n    '{name}': {type} ".format(name=f, type=format_type(self[f].dtype))
                          for f in self.col_key) if self.col_key else '\n    None'

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
            'Column key:{ck}\n' \
            'Row key:{rk}\n' \
            'Partition key:{pk}\n' \
            '----------------------------------------'.format(g=global_fields,
                                                              rk=row_key,
                                                              pk=partition_key,
                                                              r=row_fields,
                                                              ck=col_key,
                                                              c=col_fields,
                                                              e=entry_fields)
        print(s)

    @typecheck_method(order=listof(str))
    def reorder_columns(self, order: Sequence[str]) -> 'MatrixTable':
        """Reorder columns.

        .. include:: _templates/req_tstring.rst

        Examples
        --------

        Randomly shuffle order of columns:

        >>> import random
        >>> new_sample_order = [x.s for x in dataset.cols().select("s").collect()]
        >>> random.shuffle(new_sample_order)
        >>> dataset_reordered = dataset.reorder_columns(new_sample_order)

        Notes
        -----

        This method requires the keys to be unique. `order` must contain the
        same set of keys as
        ``[x.s for x in dataset.cols_table().select("s").collect()]``. The
        order of the keys in `order` determines the column order in the
        output dataset.

        Parameters
        ----------
        order : :obj:`list` of :obj:`str`
            New ordering of column keys.

        Returns
        -------
        :class:`.MatrixTable`
            Matrix table with columns reordered.
        """
        jvds = self._jvds.reorderCols(order)
        return MatrixTable(jvds)

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
        return self._jvds.nPartitions()

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
        jvds = self._jvds.coalesce(n_partitions, shuffle)
        return MatrixTable(jvds)

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
        return MatrixTable(self._jvds.naiveCoalesce(max_partitions))

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
        return MatrixTable(self._jvds.persist(storage_level))

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
        return MatrixTable(self._jvds.unpersist())

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
        return MatrixTable(self._jvds.indexRows(name))

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
        return MatrixTable(self._jvds.indexCols(name))

    @typecheck_method(other=matrix_table_type,
                      tolerance=numeric)
    def _same(self, other, tolerance=1e-6):
        return self._jvds.same(other._jvds, tolerance)

    @typecheck(datasets=matrix_table_type)
    def union_rows(*datasets: Tuple['MatrixTable']) -> 'MatrixTable':
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
            return MatrixTable(Env.hail().variant.MatrixTable.unionRows([d._jvds for d in datasets]))

    @typecheck_method(other=matrix_table_type)
    def union_cols(self, other: 'MatrixTable') -> 'MatrixTable':
        """Take the union of dataset columns.

        Examples
        --------

        .. testsetup::

            dataset_to_union_1 = dataset
            dataset_to_union_2 = dataset

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
        return MatrixTable(self._jvds.unionCols(other._jvds))

    @typecheck_method(n=int)
    def head(self, n: int) -> 'MatrixTable':
        """Subset matrix to first `n` rows.

        Examples
        --------
        Subset to the first three rows of the matrix:

        .. doctest::

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

        return MatrixTable(self._jvds.head(n))

    @typecheck_method(parts=listof(int), keep=bool)
    def _filter_partitions(self, parts, keep=True):
        return MatrixTable(self._jvds.filterPartitions(parts, keep))

    @classmethod
    @typecheck_method(table=Table, partition_key=nullable(oneof(str, listof(str))))
    def from_rows_table(cls, table: Table, partition_key: Optional[Union[str, List[str]]] = None) -> 'MatrixTable':
        """Construct matrix table with no columns from a table.

        .. include:: _templates/experimental.rst

        Examples
        --------
        Import a text table and construct a rows-only matrix table:

        >>> table = hl.import_table('data/variant-lof.tsv')
        >>> table = table.transmute(**hl.parse_variant(table['v'])).key_by('locus', 'alleles')
        >>> sites_vds = hl.MatrixTable.from_rows_table(table, partition_key='locus')

        Notes
        -----
        All fields in the table become row-indexed fields in the
        result.

        Parameters
        ----------
        table : :class:`.Table`
            The table to be converted.
        partition_key : :obj:`str` or :obj:`list` of :obj:`str`
            Partition key field(s), must be a prefix of the table key.

        Returns
        -------
        :class:`.MatrixTable`
        """
        if partition_key is not None:
            if isinstance(partition_key, str):
                partition_key = [partition_key]
            if len(partition_key) == 0:
                raise ValueError('partition_key must not be empty')
            elif list(table.key)[:len(partition_key)] != partition_key:
                raise ValueError('partition_key must be a prefix of table key')
        jmt = scala_object(Env.hail().variant, 'MatrixTable').fromRowsTable(table._jt, partition_key)
        return MatrixTable(jmt)

    @typecheck_method(p=numeric,
                      seed=int)
    def sample_rows(self, p: float, seed: int = 0) -> 'MatrixTable':
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

        return MatrixTable(self._jvds.sampleRows(p, seed))

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

        return MatrixTable(self._jvds.renameFields(row_map, col_map, entry_map, global_map))


matrix_table_type.set(MatrixTable)