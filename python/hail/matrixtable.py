from __future__ import print_function  # Python 2 and 3 print compatibility

from hail.expr.expression import *
from hail.utils import storage_level
from hail.utils.java import handle_py4j, escape_id
from hail.utils.misc import get_nice_attr_error, get_nice_field_error, wrap_to_tuple
from hail.table import Table


class GroupedMatrixTable(object):
    """Matrix table grouped by row or column that can be aggregated to produce a new matrix table.

    There are only two operations on a grouped matrix table, :meth:`.GroupedMatrixTable.partition_hint`
    and :meth:`.GroupedMatrixTable.aggregate`.

    .. testsetup::

        dataset2 = dataset.annotate_globals(global_field=5)
        table1 = dataset.rows_table()
        table1 = table1.annotate_globals(global_field=5)
        table1 = table1.annotate(consequence='SYN')

        table2 = dataset.cols_table()
        table2 = table2.annotate(pop='AMR', is_case=False, sex='F')

    """

    def __init__(self, parent, groups, grouped_indices):
        self._parent = parent
        self._groups = groups
        self._grouped_indices = grouped_indices
        self._partitions = None
        self._partition_key = None
        self._fields = {}

        for f in parent._fields:
            self._set_field(f, parent._fields[f])

    @typecheck_method(item=strlike)
    def _get_field(self, item):
        if item in self._fields:
            return self._fields[item]
        else:
            raise KeyError(get_nice_field_error(self, item))

    @typecheck_method(item=strlike)
    def __getitem__(self, item):
        return self._get_field(item)

    def __getattr__(self, item):
        if item in self.__dict__:
            return self.__dict__[item]
        else:
            raise AttributeError(get_nice_attr_error(self, item))

    def partition_by(self, *fields):
        """Set the partition key.

        Parameters
        ----------
        fields : varargs of :obj:`str`
            Fields to partition by. Must be a prefix of the key.

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
                ', '.join("'{}'".format(name) for name, _ in self._groups)))

    def partition_hint(self, n):
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

    def _set_field(self, key, value):
        assert key not in self._fields, key
        self._fields[key] = value
        if key in dir(self):
            warn("Name collision: field '{}' already in object dict."
                 " This field must be referenced with indexing syntax".format(key))
        else:
            self.__dict__[key] = value

    @handle_py4j
    def aggregate(self, **named_exprs):
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
        named_exprs = {k: to_expr(v) for k, v in named_exprs.items()}

        strs = []

        base, cleanup = self._parent._process_joins(*([x for _, x in self._groups] + named_exprs.values()))
        for k, v in named_exprs.items():
            analyze('GroupedMatrixTable.aggregate', v, self._grouped_indices,
                    {self._parent._row_axis, self._parent._col_axis})
            replace_aggregables(v._ast, 'gs')
            strs.append('{} = {}'.format(escape_id(k), v._ast.to_hql()))

        key_strs = ['{} = {}'.format(escape_id(id), e._ast.to_hql()) for id, e in self._groups]
        if self._grouped_indices == self._parent._row_indices:
            # group rows
            return cleanup(MatrixTable(base._jvds.groupVariantsBy(','.join(key_strs), ',\n'.join(strs), self._partition_key)))
        else:
            assert self._grouped_indices == self._parent._col_indices
            # group cols
            return cleanup(MatrixTable(base._jvds.groupSamplesBy(','.join(key_strs), ',\n'.join(strs))))


matrix_table_type = lazy()


class MatrixTable(object):
    """Hail's distributed implementation of a structured matrix.

    **Examples**

    .. testsetup::

        dataset2 = dataset.annotate_globals(global_field=5)
        table1 = dataset.rows_table()
        table1 = table1.annotate_globals(global_field=5)
        table1 = table1.annotate(consequence='SYN')

        table2 = dataset.cols_table()
        table2 = table2.annotate(pop='AMR', is_case=False, sex='F')

    Add annotations:

    >>> dataset = dataset.annotate_globals(pli={'SCN1A': 0.999, 'SONIC': 0.014},
    ...                                    populations = ['AFR', 'EAS', 'EUR', 'SAS', 'AMR', 'HIS'])

    >>> dataset = dataset.annotate_cols(pop = dataset.populations[functions.rand_unif(0, 6).to_int32()],
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

    >>> col_stats = dataset.aggregate_cols(pop_counts = agg.counter(dataset.pop),
    ...                                    high_quality = agg.fraction((dataset.sample_gq > 10) & (dataset.sample_dp > 5)))
    >>> print(col_stats.pop_counts)
    >>> print(col_stats.high_quality)

    >>> row_stats = dataset.aggregate_rows(het_dist = agg.stats(dataset.sas_hets))
    >>> print(row_stats.het_dist)

    >>> entry_stats = dataset.aggregate_entries(call_rate = agg.fraction(functions.is_defined(dataset.GT)),
    ...                                         global_gq_mean = agg.mean(dataset.GQ))
    >>> print(entry_stats.call_rate)
    >>> print(entry_stats.global_gq_mean)
    """

    def __init__(self, jvds):
        self._jvds = jvds

        self._globals = None
        self._col_values = None
        self._col_key = None
        self._row_key = None
        self._row_partition_key = None
        self._global_schema = None
        self._col_schema = None
        self._row_schema = None
        self._entry_schema = None
        self._num_samples = None
        self._row_axis = 'row'
        self._col_axis = 'column'
        self._global_indices = Indices(self, set())
        self._row_indices = Indices(self, {self._row_axis})
        self._col_indices = Indices(self, {self._col_axis})
        self._entry_indices = Indices(self, {self._row_axis, self._col_axis})
        self._fields = {}

        assert isinstance(self.global_schema, TStruct), self.col_schema
        assert isinstance(self.col_schema, TStruct), self.col_schema
        assert isinstance(self.row_schema, TStruct), self.row_schema
        assert isinstance(self.entry_schema, TStruct), self.entry_schema

        for f in self.global_schema.fields:
            self._set_field(f.name, construct_reference(f.name, f.typ, self._global_indices, prefix='global'))

        for f in self.col_schema.fields:
            self._set_field(f.name, construct_reference(f.name, f.typ, self._col_indices, prefix='sa'))

        for f in self.row_schema.fields:
            self._set_field(f.name, construct_reference(f.name, f.typ, self._row_indices, prefix='va'))

        for f in self.entry_schema.fields:
            self._set_field(f.name, construct_reference(f.name, f.typ, self._entry_indices, prefix='g'))

    def _set_field(self, key, value):
        assert key not in self._fields, key
        self._fields[key] = value
        if key in self.__dict__:
            warn("Name collision: field '{}' already in object dict."
                 " This field must be referenced with indexing syntax".format(key))
        else:
            self.__dict__[key] = value

    @typecheck_method(item=strlike)
    def _get_field(self, item):
        if item in self._fields:
            return self._fields[item]
        else:
            raise LookupError(get_nice_field_error(self, item))

    def __delattr__(self, item):
        if not item[0] == '_':
            raise NotImplementedError('MatrixTable objects are not mutable')

    def __setattr__(self, key, value):
        if not key[0] == '_':
            raise NotImplementedError('MatrixTable objects are not mutable')
        self.__dict__[key] = value

    def __getattr__(self, item):
        if item in self.__dict__:
            return self.__dict__[item]
        else:
            raise AttributeError(get_nice_attr_error(self, item))

    @typecheck_method(item=oneof(strlike, sized_tupleof(oneof(slice, Expression, tupleof(Expression)),
                                                        oneof(slice, Expression, tupleof(Expression)))))
    def __getitem__(self, item):
        if isinstance(item, str) or isinstance(item, unicode):
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
                return self.view_join_entries(row_key, col_key)
            elif row_key is not None and col_key is None:
                return self.view_join_rows(*row_key)
            elif row_key is None and col_key is not None:
                return self.view_join_cols(*col_key)
            else:
                return self.view_join_globals()

    @property
    def global_schema(self):
        """The schema of global fields in the matrix.

        Returns
        -------
        :class:`.TStruct`
            Global schema.
        """
        if self._global_schema is None:
            self._global_schema = Type._from_java(self._jvds.globalType())
        return self._global_schema

    @property
    def col_key(self):
        """The list of column key fields.

        Returns
        -------
        :obj:`list` of :obj:`str`
        """
        if self._col_key is None:
            self._col_key = jiterable_to_list(self._jvds.colKey())
        return self._col_key

    @property
    def col_schema(self):
        """The schema of column-indexed fields in the matrix.

        Returns
        -------
        :class:`.TStruct`
             Column schema.
        """
        if self._col_schema is None:
            self._col_schema = Type._from_java(self._jvds.colType())
        return self._col_schema

    @property
    def row_key(self):
        """The list of row key fields.

        Returns
        -------
        :obj:`list` of :obj:`str`
        """
        if self._row_key is None:
            self._row_key = jiterable_to_list(self._jvds.rowKey())
        return self._row_key

    @property
    def partition_key(self):
        """The row partition key.

        Returns
        -------
        :obj:`list` of :obj:`str`
        """
        if self._row_partition_key is None:
            self._row_partition_key = jiterable_to_list(self._jvds.rowPartitionKey())
        return self._row_partition_key

    @property
    def row_schema(self):
        """The schema of row-indexed fields in the matrix.

        Returns
        -------
        :class:`.TStruct`
             Row schema.
        """
        if self._row_schema is None:
            self._row_schema = Type._from_java(self._jvds.rowType())
        return self._row_schema

    @property
    def entry_schema(self):
        """The schema of row-and-column-indexed fields in the matrix.

        Returns
        -------
        :class:`.TStruct`
             Entry schema.
        """
        if self._entry_schema is None:
            self._entry_schema = Type._from_java(self._jvds.entryType())
        return self._entry_schema

    @handle_py4j
    def get_globals(self):
        """Returns the global values of the dataset as Python values.

        Returns
        -------
        :class:`.Struct`
            Global values.
        """
        if self._globals is None:
            self._globals = self.global_schema._convert_to_py(self._jvds.globals())
        return self._globals

    @property
    @handle_py4j
    def globals(self):
        """Returns a struct expression including all global fields.

        Returns
        -------
        :class:`.StructExpression`
            Struct of all global fields.
        """
        return construct_expr(Reference('global', False), self.global_schema,
                              indices=self._global_indices,
                              refs=LinkedList(tuple).push(
                                  *[(f.name, self._global_indices) for f in self.global_schema.fields]))

    @property
    @handle_py4j
    def row(self):
        """Returns a struct expression including all row-indexed fields.

        Returns
        -------
        :class:`.StructExpression`
            Struct of all row fields.
        """
        return construct_expr(Reference('va', False), self.row_schema,
                              indices=self._row_indices,
                              refs=LinkedList(tuple).push(
                                  *[(f.name, self._row_indices) for f in self.row_schema.fields]))

    @property
    @handle_py4j
    def col(self):
        """Returns a struct expression including all column-indexed fields.

        Returns
        -------
        :class:`.StructExpression`
            Struct of all column fields.
        """
        return construct_expr(Reference('sa', False), self.col_schema,
                              indices=self._col_indices,
                              refs=LinkedList(tuple).push(
                                  *[(f.name, self._col_indices) for f in self.col_schema.fields]))

    @property
    @handle_py4j
    def entry(self):
        """Returns a struct expression including all row-and-column-indexed fields.

        Returns
        -------
        :class:`.StructExpression`
            Struct of all entry fields.
        """
        return construct_expr(Reference('g', False), self.entry_schema,
                              indices=self._entry_indices,
                              refs=LinkedList(tuple).push(
                                  *[(f.name, self._entry_indices) for f in self.entry_schema.fields]))

    @typecheck_method(fields=strlike)
    def key_cols_by(self, *fields):
        """Key columns by a new set of fields.

        Parameters
        ----------
        fields : varargs of :obj:`str`
            Fields to key by.
        Returns
        -------
        :class:`.MatrixTable`
        """
        for f in fields:
            if f not in self._fields:
                raise ValueError("MatrixTable has no field '{}'".format(f))
            if not self[f]._indices == self._col_indices:
                raise ValueError("field '{}' is not a column field".format(f))
        return MatrixTable(self._jvds.keyColsBy(list(fields)))

    @typecheck_method(fields=strlike)
    def key_rows_by(self, *fields):
        """Key rows by a new set of fields.

        Parameters
        ----------
        fields : varargs of :obj:`str`
            Fields to key by.
        Returns
        -------
        :class:`.MatrixTable`
        """
        for f in fields:
            if f not in self._fields:
                raise ValueError("MatrixTable has no field '{}'".format(f))
            if not self[f]._indices == self._row_indices:
                raise ValueError("field '{}' is not a column field".format(f))
        return MatrixTable(self._jvds.keyRowsBy(list(fields), list(fields)))

    @handle_py4j
    def annotate_globals(self, **named_exprs):
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
            self._check_field_name(k, self._global_indices)
        m = MatrixTable(base._jvds.annotateGlobalExpr(",\n".join(exprs)))
        return cleanup(m)

    @handle_py4j
    def annotate_rows(self, **named_exprs):
        """Create new row-indexed fields by name.

        Examples
        --------
        Compute call statistics for high quality samples per variant:

        >>> high_quality_calls = agg.filter(dataset.sample_qc.gqMean > 20, dataset.GT)
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
            replace_aggregables(v._ast, 'gs')
            exprs.append('{k} = {v}'.format(k=escape_id(k), v=v._ast.to_hql()))
            self._check_field_name(k, self._row_indices)
        m = MatrixTable(base._jvds.annotateVariantsExpr(",\n".join(exprs)))
        return cleanup(m)

    @handle_py4j
    def annotate_cols(self, **named_exprs):
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
            replace_aggregables(v._ast, 'gs')
            exprs.append('{k} = {v}'.format(k=escape_id(k), v=v._ast.to_hql()))
            self._check_field_name(k, self._col_indices)
        m = MatrixTable(base._jvds.annotateSamplesExpr(",\n".join(exprs)))
        return cleanup(m)

    @handle_py4j
    def annotate_entries(self, **named_exprs):
        """Create new row-and-column-indexed fields by name.

        Examples
        --------
        Compute the allele dosage using the PL field:

        >>> def get_dosage(pl):
        ...    # convert to linear scale
        ...    linear_scaled = pl.map(lambda x: 10 ** - (x / 10))
        ...
        ...    # normalize to sum to 1
        ...    ls_sum = linear_scaled.sum()
        ...    linear_scaled = linear_scaled.map(lambda x: x / ls_sum)
        ...
        ...    # multiply by [0, 1, 2] and sum
        ...    return (linear_scaled * [0, 1, 2]).sum()
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
            self._check_field_name(k, self._entry_indices)
        m = MatrixTable(base._jvds.annotateGenotypesExpr(",\n".join(exprs)))
        return cleanup(m)

    @handle_py4j
    def select_globals(self, *exprs, **named_exprs):
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
        exprs = [to_expr(e) if not isinstance(e, str) and not isinstance(e, unicode) else self[e] for e in exprs]
        named_exprs = {k: to_expr(v) for k, v in named_exprs.items()}
        strs = []
        all_exprs = []
        base, cleanup = self._process_joins(*(exprs + named_exprs.values()))
        for e in exprs:
            all_exprs.append(e)
            analyze('MatrixTable.select_globals', e, self._global_indices)
            if e._ast.search(lambda ast: not isinstance(ast, Reference) and not isinstance(ast, Select)):
                raise ExpressionException("method 'select_globals' expects keyword arguments for complex expressions")
            strs.append(
                '{}: {}'.format(e._ast.selection if isinstance(e._ast, Select) else e._ast.name, e._ast.to_hql()))
        for k, e in named_exprs.items():
            all_exprs.append(e)
            analyze('MatrixTable.select_globals', e, self._global_indices)
            self._check_field_name(k, self._global_indices)
            strs.append('{}: {}'.format(escape_id(k), to_expr(e)._ast.to_hql()))
        m = MatrixTable(base._jvds.annotateGlobalExpr('global = {' + ',\n'.join(strs) + '}'))
        return cleanup(m)

    @handle_py4j
    def select_rows(self, *exprs, **named_exprs):
        """Select existing row fields or create new fields by name, dropping the rest.

        Examples
        --------
        Select existing fields and compute a new one:

        >>> dataset_result = dataset.select_rows(dataset.variant_qc.gqMean,
        ...                                      highQualityCases = agg.count_where((dataset.GQ > 20) & (dataset.isCase)))

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
        exprs = [to_expr(e) if not isinstance(e, str) and not isinstance(e, unicode) else self[e] for e in exprs]
        named_exprs = {k: to_expr(v) for k, v in named_exprs.items()}
        strs = []
        all_exprs = []
        base, cleanup = self._process_joins(*(exprs + named_exprs.values()))

        for e in exprs:
            all_exprs.append(e)
            analyze('MatrixTable.select_rows', e, self._row_indices, {self._col_axis})
            if e._ast.search(lambda ast: not isinstance(ast, Reference) and not isinstance(ast, Select)):
                raise ExpressionException("method 'select_rows' expects keyword arguments for complex expressions")
            replace_aggregables(e._ast, 'gs')
            strs.append(e._ast.to_hql())
        for k, e in named_exprs.items():
            all_exprs.append(e)
            analyze('MatrixTable.select_rows', e, self._row_indices, {self._col_axis})
            self._check_field_name(k, self._row_indices)
            replace_aggregables(e._ast, 'gs')
            strs.append('{} = {}'.format(escape_id(k), e._ast.to_hql()))
        m = MatrixTable(base._jvds.selectRows(strs))
        return cleanup(m)

    @handle_py4j
    def select_cols(self, *exprs, **named_exprs):
        """Select existing column fields or create new fields by name, dropping the rest.

        Examples
        --------
        Select existing fields and compute a new one:

        >>> dataset_result = dataset.select_cols(dataset.sample_qc,
        ...                                      dataset.pheno.age,
        ...                                      isCohort1 = dataset.pheno.cohortName == 'Cohort1')

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

        exprs = [to_expr(e) if not isinstance(e, str) and not isinstance(e, unicode) else self[e] for e in exprs]
        named_exprs = {k: to_expr(v) for k, v in named_exprs.items()}
        strs = []
        all_exprs = []
        base, cleanup = self._process_joins(*(exprs + named_exprs.values()))

        for e in exprs:
            all_exprs.append(e)
            analyze('MatrixTable.select_cols', e, self._col_indices, {self._row_axis})
            if e._ast.search(lambda ast: not isinstance(ast, Reference) and not isinstance(ast, Select)):
                raise ExpressionException("method 'select_cols' expects keyword arguments for complex expressions")
            replace_aggregables(e._ast, 'gs')
            strs.append(e._ast.to_hql())
        for k, e in named_exprs.items():
            all_exprs.append(e)
            analyze('MatrixTable.select_cols', e, self._col_indices, {self._row_axis})
            self._check_field_name(k, self._col_indices)
            replace_aggregables(e._ast, 'gs')
            strs.append('{} = {}'.format(escape_id(k), e._ast.to_hql()))

        m = MatrixTable(base._jvds.selectCols(strs))
        return cleanup(m)

    @handle_py4j
    def select_entries(self, *exprs, **named_exprs):
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
        exprs = [to_expr(e) if not isinstance(e, str) and not isinstance(e, unicode) else self[e] for e in exprs]
        named_exprs = {k: to_expr(v) for k, v in named_exprs.items()}
        strs = []
        all_exprs = []
        base, cleanup = self._process_joins(*(exprs + named_exprs.values()))

        for e in exprs:
            all_exprs.append(e)
            analyze('MatrixTable.select_entries', e, self._entry_indices)
            if e._ast.search(lambda ast: not isinstance(ast, Reference) and not isinstance(ast, Select)):
                raise ExpressionException("method 'select_entries' expects keyword arguments for complex expressions")
            strs.append(e._ast.to_hql())
        for k, e in named_exprs.items():
            all_exprs.append(e)
            analyze('MatrixTable.select_entries', e, self._entry_indices)
            self._check_field_name(k, self._entry_indices)
            strs.append('{} = {}'.format(escape_id(k), e._ast.to_hql()))
        m = MatrixTable(base._jvds.selectEntries(strs))
        return cleanup(m)

    @handle_py4j
    @typecheck_method(exprs=oneof(strlike, Expression))
    def drop(self, *exprs):
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
                assert isinstance(e, str) or isinstance(str, unicode)
                if e not in self._fields:
                    raise IndexError("matrix has no field '{}'".format(e))
                fields_to_drop.add(e)

        m = self
        if any(self._fields[field]._indices == self._global_indices for field in fields_to_drop):
            # need to drop globals
            new_global_fields = [k.name for k in m.global_schema.fields if k.name not in fields_to_drop]
            m = m.select_globals(*new_global_fields)

        row_fields = [x for x in fields_to_drop if self._fields[x]._indices == self._row_indices]
        if row_fields:
            # need to drop row fields
            m = MatrixTable(m._jvds.dropRows(row_fields))

        if any(self._fields[field]._indices == self._col_indices for field in fields_to_drop):
            # need to drop col fields
            new_col_fields = [k.name for k in m.col_schema.fields if k.name not in fields_to_drop]
            m = m.select_cols(*new_col_fields)

        entry_fields = [x for x in fields_to_drop if self._fields[x]._indices == self._entry_indices]
        if any(self._fields[field]._indices == self._entry_indices for field in fields_to_drop):
            # need to drop entry fields
            m = MatrixTable(m._jvds.dropEntries(entry_fields))

        return m

    @handle_py4j
    def drop_rows(self):
        """Drop all rows of the matrix.  Is equivalent to:

        >>> dataset_result = dataset.filter_rows(False)

        Returns
        -------
        :class:`.MatrixTable`
            Matrix table with no rows.
        """
        warn("deprecation: 'drop_rows' will be removed before 0.2 release")
        return MatrixTable(self._jvds.dropVariants())

    def drop_cols(self):
        """Drop all columns of the matrix.  Is equivalent to:

        >>> dataset_result = dataset.filter_cols(False)

        Returns
        -------
        :class:`.MatrixTable`
            Matrix table with no columns.
        """
        warn("deprecation: 'drop_cols' will be removed before 0.2 release")
        return MatrixTable(self._jvds.dropSamples())

    @handle_py4j
    @typecheck_method(expr=anytype, keep=bool)
    def filter_rows(self, expr, keep=True):
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
        expr = to_expr(expr)
        base, cleanup = self._process_joins(expr)
        analyze('MatrixTable.filter_rows', expr, self._row_indices, {self._col_axis})
        replace_aggregables(expr._ast, 'gs')
        m = MatrixTable(base._jvds.filterVariantsExpr(expr._ast.to_hql(), keep))
        return cleanup(m)

    @handle_py4j
    @typecheck_method(expr=anytype, keep=bool)
    def filter_cols(self, expr, keep=True):
        """Filter columns of the matrix.

        Examples
        --------

        Keep columns where `pheno.isCase` is ``True`` and `pheno.age` is larger than 50:

        >>> dataset_result = dataset.filter_cols(dataset.pheno.isCase & (dataset.pheno.age > 50), keep=True)

        Remove rows where `sample_qc.gqMean` is less than 20:

        >>> dataset_result = dataset.filter_cols(dataset.sample_qc.gqMean < 20, keep=False)

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

        replace_aggregables(expr._ast, 'gs')
        m = MatrixTable(base._jvds.filterSamplesExpr(expr._ast.to_hql(), keep))
        return cleanup(m)

    @handle_py4j
    def filter_entries(self, expr, keep=True):
        """Filter entries of the matrix.

        Examples
        --------

        Keep entries where the sum of `AD` is greater than 10 and `GQ` is greater than 20:

        >>> dataset_result = dataset.filter_entries((dataset.AD.sum() > 10) & (dataset.GQ > 20))

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

        m = MatrixTable(base._jvds.filterGenotypes(expr._ast.to_hql(), keep))
        return cleanup(m)

    def transmute_globals(self, **named_exprs):
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

    def transmute_rows(self, **named_exprs):
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

    @handle_py4j
    def transmute_cols(self, **named_exprs):
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

    @handle_py4j
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

    @handle_py4j
    def aggregate_rows(self, **named_exprs):
        """Aggregate over rows into a local struct.

        Examples
        --------
        Aggregate over rows:

        .. doctest::

            >>> dataset.aggregate_rows(n_high_quality=agg.count_where(dataset.qual > 40),
            ...                        mean_qual = agg.mean(dataset.qual))
            Struct(n_high_quality=100150224, mean_qual=50.12515572)

        Notes
        -----
        Unlike most :class:`.MatrixTable` methods, this method does not support
        meaningful references to fields that are not global or indexed by row.

        This method should be thought of as a more convenient alternative to
        the following:

        >>> rows_table = dataset.rows_table()
        >>> rows_table.aggregate(n_high_quality=agg.count_where(rows_table.qual > 40),
        ...                      mean_qual = agg.mean(rows_table.qual))

        Note
        ----
        This method supports (and expects!) aggregation over rows.

        Parameters
        ----------
        named_exprs : keyword args of :class:`.Expression`
            Aggregation expressions.

        Returns
        -------
        :class:`.Struct`
            Struct containing all results.
        """

        str_exprs = []
        named_exprs = {k: to_expr(v) for k, v in named_exprs.items()}
        base, _ = self._process_joins(*named_exprs.values())

        for k, v in named_exprs.items():
            analyze('MatrixTable.aggregate_rows', v, self._global_indices, {self._row_axis})
            replace_aggregables(v._ast, 'variants')
            str_exprs.append(v._ast.to_hql())

        result_list = self._jvds.queryVariants(jarray(Env.jvm().java.lang.String, str_exprs))
        ptypes = [Type._from_java(x._2()) for x in result_list]

        annotations = [ptypes[i]._convert_to_py(result_list[i]._1()) for i in range(len(ptypes))]
        d = {k: v for k, v in zip(named_exprs.keys(), annotations)}
        return Struct(**d)

    @handle_py4j
    def aggregate_cols(self, **named_exprs):
        """Aggregate over columns into a local struct.

        Examples
        --------
        Aggregate over columns:

        .. doctest::

            >>> dataset.aggregate_cols(fraction_female=agg.fraction(dataset.pheno.isFemale),
            ...                        case_ratio = agg.count_where(dataset.isCase) / agg.count())
            Struct(fraction_female=0.5102222, case_ratio=0.35156)

        Notes
        -----
        Unlike most :class:`.MatrixTable` methods, this method does not support
        meaningful references to fields that are not global or indexed by column.

        This method should be thought of as a more convenient alternative to
        the following:

        >>> cols_table = dataset.cols_table()
        >>> cols_table.aggregate(fraction_female=agg.fraction(cols_table.pheno.isFemale),
        ...                      case_ratio = agg.count_where(cols_table.isCase) / agg.count())

        Note
        ----
        This method supports (and expects!) aggregation over columns.

        Parameters
        ----------
        named_exprs : keyword args of :class:`.Expression`
            Aggregation expressions.

        Returns
        -------
        :class:`.Struct`
            Struct containing all results.
        """

        str_exprs = []
        named_exprs = {k: to_expr(v) for k, v in named_exprs.items()}
        base, _ = self._process_joins(*named_exprs.values())

        for k, v in named_exprs.items():
            analyze('MatrixTable.aggregate_cols', v, self._global_indices, {self._col_axis})
            replace_aggregables(v._ast, 'samples')
            str_exprs.append(v._ast.to_hql())

        result_list = base._jvds.querySamples(jarray(Env.jvm().java.lang.String, str_exprs))
        ptypes = [Type._from_java(x._2()) for x in result_list]

        annotations = [ptypes[i]._convert_to_py(result_list[i]._1()) for i in range(len(ptypes))]
        d = {k: v for k, v in zip(named_exprs.keys(), annotations)}
        return Struct(**d)

    @handle_py4j
    def aggregate_entries(self, **named_exprs):
        """Aggregate over all entries into a local struct.

        Examples
        --------
        Aggregate over entries:

        .. doctest::

            >>> dataset.aggregate_entries(global_gq_mean = agg.mean(dataset.GQ),
            ...                           call_rate = agg.fraction(functions.is_defined(dataset.GT)))
            Struct(global_gq_mean=31.16200, call_rate=0.981682)

        Notes
        -----
        This method should be thought of as a more convenient alternative to
        the following:

        >>> entries_table = dataset.entries_table()
        >>> entries_table.aggregate(global_gq_mean = agg.mean(entries_table.GQ),
        ...                         call_rate = agg.fraction(functions.is_defined(entries_table.GT)))

        Note
        ----
        This method supports (and expects!) aggregation over entries.

        Parameters
        ----------
        named_exprs : keyword args of :class:`.Expression`
            Aggregation expressions.

        Returns
        -------
        :class:`.Struct`
            Struct containing all results.
        """

        str_exprs = []
        named_exprs = {k: to_expr(v) for k, v in named_exprs.items()}
        base, _ = self._process_joins(*named_exprs.values())

        for k, v in named_exprs.items():
            analyze('MatrixTable.aggregate_entries', v, self._global_indices, {self._row_axis, self._col_axis})
            replace_aggregables(v._ast, 'gs')
            str_exprs.append(v._ast.to_hql())

        result_list = base._jvds.queryGenotypes(jarray(Env.jvm().java.lang.String, str_exprs))
        ptypes = [Type._from_java(x._2()) for x in result_list]

        annotations = [ptypes[i]._convert_to_py(result_list[i]._1()) for i in range(len(ptypes))]
        d = {k: v for k, v in zip(named_exprs.keys(), annotations)}
        return Struct(**d)

    @handle_py4j
    def explode_rows(self, field_expr):
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
        if isinstance(field_expr, str) or isinstance(field_expr, unicode):
            if not field_expr in self._fields:
                raise KeyError("MatrixTable has no field '{}'".format(field_expr))
            elif self._fields[field_expr]._indices != self._row_indices:
                raise ExpressionException("Method 'explode_rows' expects a field indexed by row, found axes '{}'"
                                          .format(self._fields[field_expr]._indices.axes))
            s = 'va.{}'.format(escape_id(field_expr))
        else:
            e = to_expr(field_expr)
            analyze('MatrixTable.explode_rows', field_expr, self._row_indices, set(self._fields.keys()))
            if e._ast.search(lambda ast: not isinstance(ast, Reference) and not isinstance(ast, Select)):
                raise ExpressionException(
                    "method 'explode_rows' requires a field or subfield, not a complex expression")
            s = e._ast.to_hql()
        return MatrixTable(self._jvds.explodeVariants(s))

    @handle_py4j
    def explode_cols(self, field_expr):
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

        if isinstance(field_expr, str) or isinstance(field_expr, unicode):
            if not field_expr in self._fields:
                raise KeyError("MatrixTable has no field '{}'".format(field_expr))
            elif self._fields[field_expr]._indices != self._col_indices:
                raise ExpressionException("Method 'explode_cols' expects a field indexed by col, found axes '{}'"
                                          .format(self._fields[field_expr]._indices.axes))
            s = 'sa.{}'.format(escape_id(field_expr))
        else:
            e = to_expr(field_expr)
            analyze('MatrixTable.explode_cols', field_expr, self._col_indices)
            if e._ast.search(lambda ast: not isinstance(ast, Reference) and not isinstance(ast, Select)):
                raise ExpressionException(
                    "method 'explode_cols' requires a field or subfield, not a complex expression")
            s = e._ast.to_hql()
        return MatrixTable(self._jvds.explodeSamples(s))

    @handle_py4j
    def group_rows_by(self, *exprs, **named_exprs):
        """Group rows, used with :meth:`.GroupedMatrixTable.aggregate`

        Examples
        --------
        Aggregate to a matrix with genes as row keys, computing the number of
        non-reference calls as an entry field:

        >>> dataset_result = (dataset.group_rows_by(dataset.gene)
        ...                          .aggregate(n_non_ref = agg.count_where(dataset.GT.is_non_ref())))

        Notes
        -----
        The `key_expr` argument can either be a string referring to a row-indexed field
        of the dataset, or an expression that will become the new row key.

        Parameters
        ----------
        key_expr : str or :class:`.Expression`
            Field name or expression to use as new row key.

        Returns
        -------
        :class:`.GroupedMatrixTable`
            Grouped matrix, can be used to call :meth:`.GroupedMatrixTable.aggregate`.
        """
        groups = []
        for e in exprs:
            if isinstance(e, str) or isinstance(e, unicode):
                e = self[e]
            else:
                e = to_expr(e)
            analyze('MatrixTable.group_rows_by', e, self._row_indices)
            ast = e._ast.expand()
            if any(not isinstance(a, Reference) and not isinstance(a, Select) for a in ast):
                raise ExpressionException("method 'group_rows_by' expects keyword arguments for complex expressions")
            key = ast[0].name if isinstance(ast[0], Reference) else ast[0].selection
            groups.append((key, e))
        for k, e in named_exprs.items():
            e = to_expr(e)
            analyze('MatrixTable.group_rows_by', e, self._row_indices)
            groups.append((k, e))

        return GroupedMatrixTable(self, groups, self._row_indices)

    @handle_py4j
    def group_cols_by(self, *exprs, **named_exprs):
        """Group rows, used with :meth:`.GroupedMatrixTable.aggregate`

        Examples
        --------
        Aggregate to a matrix with cohort as column keys, computing the call rate
        as an entry field:

        .. testsetup::

            dataset = dataset.annotate_cols(cohort = 'cohort')

        >>> dataset_result = (dataset.group_cols_by(dataset.cohort)
        ...                          .aggregate(call_rate = agg.fraction(functions.is_defined(dataset.GT))))

        Notes
        -----
        The `key_expr` argument can either be a string referring to a column-indexed
        field of the dataset, or an expression that will become the new column key.

        Parameters
        ----------
        key_expr : str or :class:`.Expression`
            Field name or expression to use as new column key.

        Returns
        -------
        :class:`.GroupedMatrixTable`
            Grouped matrix, can be used to call :meth:`.GroupedMatrixTable.aggregate`.
        """
        groups = []
        for e in exprs:
            if isinstance(e, str) or isinstance(e, unicode):
                e = self[e]
            else:
                e = to_expr(e)
            analyze('MatrixTable.group_cols_by', e, self._col_indices)
            ast = e._ast.expand()
            if any(not isinstance(a, Reference) and not isinstance(a, Select) for a in ast):
                raise ExpressionException("method 'group_rows_by' expects keyword arguments for complex expressions")
            key = ast[0].name if isinstance(ast[0], Reference) else ast[0].selection
            groups.append((key, e))
        for k, e in named_exprs.items():
            e = to_expr(e)
            analyze('MatrixTable.group_cols_by', e, self._col_indices)
            groups.append((k, e))

        return GroupedMatrixTable(self, groups, self._col_indices)

    @handle_py4j
    def count_rows(self):
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
        return self._jvds.countVariants()

    @handle_py4j
    def count_cols(self):
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

    @handle_py4j
    def count(self):
        """Count the number of columns and rows in the matrix.

        Examples
        --------
        .. doctest::

            >>> dataset.count_cols()

        Returns
        -------
        :obj:`int`, :obj:`int`
            Number of cols, number of rows.
        """
        r = self._jvds.count()
        return r._1(), r._2()

    @handle_py4j
    @typecheck_method(output=strlike,
                      overwrite=bool)
    def write(self, output, overwrite=False):
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

        self._jvds.write(output, overwrite)

    @handle_py4j
    def rows_table(self):
        """Returns a table with all row fields in the matrix.

        Examples
        --------
        Extract the row table:

        >>> rows_table = dataset.rows_table()

        Returns
        -------
        :class:`.Table`
            Table with all row fields from the matrix, with one row per row of the matrix.
        """
        return Table(self._jvds.rowsTable())

    @handle_py4j
    def cols_table(self):
        """Returns a table with all column fields in the matrix.

        Examples
        --------
        Extract the column table:

        >>> cols_table = dataset.cols_table()

        Returns
        -------
        :class:`.Table`
            Table with all column fields from the matrix, with one row per column of the matrix.
        """
        return Table(self._jvds.colsTable())

    @handle_py4j
    def entries_table(self):
        """Returns a matrix in coordinate table form.

        Examples
        --------
        Extract the entry table:

        >>> entries_table = dataset.entries_table()

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

    @handle_py4j
    def view_join_globals(self):
        uid = Env._get_uid()

        def joiner(obj):
            if isinstance(obj, MatrixTable):
                return MatrixTable(Env.jutils().joinGlobals(obj._jvds, self._jvds, uid))
            else:
                assert isinstance(obj, Table)
                return Table(Env.jutils().joinGlobals(obj._jt, self._jvds, uid))

        return construct_expr(GlobalJoinReference(uid), self.global_schema,
                              joins=LinkedList(Join).push(Join(joiner, [uid], uid)))

    @handle_py4j
    def view_join_rows(self, *exprs):
        exprs = [to_expr(e) for e in exprs]
        indices, aggregations, joins, refs = unify_all(*exprs)
        src = indices.source

        if aggregations:
            raise ExpressionException('Cannot join using an aggregated field')
        uid = Env._get_uid()
        uids_to_delete = [uid]

        if src is None:
            raise ExpressionException('Cannot index with a scalar expression')

        if isinstance(src, Table):
            # join table with matrix.rows_table()
            right = self.rows_table()
            return right.view_join_rows(*exprs)
        else:
            assert isinstance(src, MatrixTable)
            right = self

            # fast path
            is_row_key = len(exprs) == len(src.row_key) and all(
                exprs[i] is src._fields[src.row_key[i]] for i in range(len(exprs)))
            is_partition_key = len(exprs) == len(src.partition_key) and all(
                exprs[i] is src._fields[src.partition_key[i]] for i in range(len(exprs)))

            if is_row_key or is_partition_key:
                prefix = 'va'
                joiner = lambda left: (
                    MatrixTable(left._jvds.annotateVariantsVDS(right._jvds, uid)))
            else:
                return self.rows_table().view_join_rows(*exprs)

            schema = TStruct.from_fields([f for f in self.row_schema.fields if f.name not in self.row_key])
            return construct_expr(Select(Reference(prefix), uid),
                                  schema, indices, aggregations,
                                  joins.push(Join(joiner, uids_to_delete, uid)), refs)

    @handle_py4j
    def view_join_cols(self, *exprs):
        exprs = [to_expr(e) for e in exprs]
        indices, aggregations, joins, refs = unify_all(*exprs)
        src = indices.source

        if aggregations:
            raise ExpressionException('Cannot join using an aggregated field')
        uid = Env._get_uid()
        uids_to_delete = [uid]

        if src is None:
            raise ExpressionException('Cannot index with a scalar expression')

        return self.cols_table().view_join_rows(*exprs)

    @handle_py4j
    def view_join_entries(self, row_exprs, col_exprs):
        row_exprs = [to_expr(e) for e in row_exprs]
        col_exprs = [to_expr(e) for e in col_exprs]

        indices, aggregations, joins, refs = unify_all(*(row_exprs + col_exprs))
        src = indices.source
        if aggregations:
            raise ExpressionException('Cannot join using an aggregated field')
        uid = Env._get_uid()
        uids_to_delete = [uid]

        if isinstance(src, Table):
            # join table with matrix.entries_table()
            return self.entries_table().view_join_rows(*(row_exprs + col_exprs))
        else:
            raise NotImplementedError('matrix.view_join_entries with {}'.format(src.__class__))

    @typecheck_method(name=strlike, indices=Indices)
    def _check_field_name(self, name, indices):
        if name in set(self._fields.keys()) and not self._fields[name]._indices == indices:
            msg = 'name collision with field indexed by {}: {}'.format(indices, name)
            error('Analysis exception: {}'.format(msg))
            raise ExpressionException(msg)

    @typecheck_method(exprs=Expression)
    def _process_joins(self, *exprs):

        all_uids = []
        left = self
        used_uids = set()

        for e in exprs:
            rewrite_global_refs(e._ast, self)
            for j in list(e._joins)[::-1]:
                if j.uid not in used_uids:
                    left = j.join_function(left)
                    all_uids.extend(j.temp_vars)
                    used_uids.add(j.uid)

        def cleanup(matrix):
            remaining_uids = [uid for uid in all_uids if uid in matrix._fields]
            return matrix.drop(*remaining_uids)

        return left, cleanup

    @typecheck_method(truncate_at=integral)
    def describe(self, truncate_at=60):
        """Print information about the fields in the matrix."""

        def format_type(typ):
            return typ.pretty(indent=4)

        if len(self.global_schema.fields) == 0:
            global_fields = '\n    None'
        else:
            global_fields = ''.join("\n    '{name}': {type} ".format(
                name=fd.name, type=format_type(fd.typ)) for fd in self.global_schema.fields)

        row_fields = ''.join("\n    '{name}': {type} ".format(
            name=fd.name, type=format_type(fd.typ)) for fd in self.row_schema.fields)

        row_key = ''.join("\n    '{name}': {type} ".format(name=f, type=format_type(self[f].dtype))
                          for f in self.row_key) if self.row_key else '\n    None'
        partition_key = ''.join("\n    '{name}': {type} ".format(name=f, type=format_type(self[f].dtype))
                                for f in self.partition_key) if self.partition_key else '\n    None'

        col_fields = ''.join("\n    '{name}': {type} ".format(
            name=fd.name, type=format_type(fd.typ)) for fd in self.col_schema.fields)

        col_key = ''.join("\n    '{name}': {type} ".format(name=f, type=format_type(self[f].dtype))
                          for f in self.col_key) if self.col_key else '\n    None'

        if len(self.entry_schema.fields) == 0:
            entry_fields = '\n    None'
        else:
            entry_fields = ''.join("\n    '{name}': {type} ".format(
                name=fd.name, type=format_type(fd.typ)) for fd in self.entry_schema.fields)

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

    @handle_py4j
    @typecheck_method(order=listof(strlike))
    def reorder_columns(self, order):
        """Reorder columns.

        .. include:: _templates/req_tstring.rst

        Examples
        --------

        Randomly shuffle order of columns:

        >>> import random
        >>> new_sample_order = [x.s for x in dataset.cols_table().select("s").collect()]
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
        jvds = self._jvds.reorderSamples(order)
        return MatrixTable(jvds)

    @handle_py4j
    def num_partitions(self):
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

    @handle_py4j
    @typecheck_method(num_partitions=integral,
                      shuffle=bool)
    def repartition(self, num_partitions, shuffle=True):
        """Increase or decrease the number of partitions.

        Examples
        --------

        Repartition to 500 partitions:

        >>> dataset_result = dataset.repartition(500)

        Notes
        -----

        Check the current number of partitions with :meth:`.num_partitions`.

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
        respectively. In particular, when ``shuffle=False``, ``num_partitions``
        cannot exceed current number of partitions.

        Note
        ----
        If `shuffle` is ``False``, the number of partitions may only be
        reduced, not increased.

        Parameters
        ----------
        num_partitions : int
            Desired number of partitions.
        shuffle : bool
            If ``True``, use full shuffle to repartition.

        Returns
        -------
        :class:`.MatrixTable`
            Repartitioned dataset.
        """
        jvds = self._jvds.coalesce(num_partitions, shuffle)
        return MatrixTable(jvds)

    @handle_py4j
    @typecheck_method(max_partitions=integral)
    def naive_coalesce(self, max_partitions):
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

    @handle_py4j
    def cache(self):
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
    def persist(self, storage_level='MEMORY_AND_DISK'):
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

    @handle_py4j
    def unpersist(self):
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

    @handle_py4j
    @typecheck_method(name=strlike)
    def index_rows(self, name='row_idx'):
        """Add the integer index of each row as a new row field.

        Examples
        --------

        >>> dataset_result = dataset.index_rows()

        Notes
        -----
        The field added is type :class:`.TInt64`.

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

    @handle_py4j
    @typecheck_method(name=strlike)
    def index_cols(self, name='col_idx'):
        """Add the integer index of each column as a new column field.

        Examples
        --------

        >>> dataset_result = dataset.index_cols()

        Notes
        -----
        The field added is type :class:`.TInt32`.

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

    @handle_py4j
    @typecheck_method(other=matrix_table_type,
                      tolerance=numeric)
    def _same(self, other, tolerance=1e-6):
        return self._jvds.same(other._jvds, tolerance)

    @handle_py4j
    @typecheck(datasets=matrix_table_type)
    def union_rows(*datasets):
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
        >>> dataset_result = MatrixTable.union_rows(*all_datasets)

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

    @handle_py4j
    @typecheck_method(other=matrix_table_type)
    def union_cols(self, other):
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

    @handle_py4j
    @typecheck_method(n=integral)
    def head(self, n):
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

    @handle_py4j
    @typecheck_method(parts=listof(integral), keep=bool)
    def _filter_partitions(self, parts, keep=True):
        return MatrixTable(self._jvds.filterPartitions(parts, keep))

    @classmethod
    @handle_py4j
    @typecheck_method(table=Table)
    def from_rows_table(cls, table):
        """Construct matrix table with no columns from a table.

        Examples
        --------
        Import a text table and construct a rows-only matrix table:

        >>> table = methods.import_table('data/variant-lof.tsv', key='v')
        >>> sites_vds = MatrixTable.from_rows_table(table)

        Notes
        -----
        The table must be keyed by a single field.

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
        jmt = scala_object(Env.hail().variant, 'MatrixTable').fromRowsTable(table._jt)
        return MatrixTable(jmt)

    @handle_py4j
    @typecheck_method(p=numeric,
                      seed=integral)
    def sample_rows(self, p, seed=0):
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
            Matrix table with approximately ``p * num_rows`` rows.
        """

        if not (0 <= p <= 1):
            raise ValueError("Requires 'p' in [0,1]. Found p={}".format(p))

        return MatrixTable(self._jvds.sampleVariants(p, seed))


matrix_table_type.set(MatrixTable)
