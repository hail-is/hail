from __future__ import print_function  # Python 2 and 3 print compatibility

from hail.expr.expression import *
from hail.utils.java import handle_py4j
from hail.api2 import Table


class GroupedMatrixTable(object):
    """Matrix table grouped by row or column that can be aggregated to produce a new matrix table.

    There are only two operations on a grouped matrix table, :meth:`GroupedMatrixTable.partition_hint`
    and :meth:`GroupedMatrixTable.aggregate`.

    .. testsetup::

        dataset2 = dataset.annotate_globals(global_field=5)
        table1 = dataset.rows_table()
        table1 = table1.annotate_globals(global_field=5)
        table1 = table1.annotate(consequence='SYN')

        table2 = dataset.cols_table()
        table2 = table2.annotate(pop='AMR', is_case=False, sex='F')

    """
    def __init__(self, parent, group, grouped_indices):
        self._parent = parent
        self._group = group
        self._grouped_indices = grouped_indices
        self._partitions = None
        self._fields = {}

        for f in parent._fields:
            self._set_field(f, parent._fields[f])

    def partition_hint(self, n):
        """Set the target number of partitions for aggregation.

        Examples
        --------

        Use `partition_hint` in a :meth:`MatrixTable.group_rows_by` /
        :meth:`GroupedMatrixTable.aggregate` pipeline:

        >>> dataset_result = (dataset.group_rows_by(dataset.gene)
        ...                          .partition_hint(5)
        ...                          .aggregate(n_non_ref = agg.count_where(dataset.GT.is_non_ref())))

        Notes
        -----
        Until Hail's query optimizer is intelligent enough to sample records at all
        stages of a pipeline, it can be necessary in some places to provide some
        explicit hints.

        The default number of partitions for :meth:`GroupedMatrixTable.aggregate` is
        the number of partitions in the upstream dataset. If the aggregation greatly
        reduces the size of the dataset, providing a hint for the target number of
        partitions can accelerate downstream operations.

        Parameters
        ----------
        n : int
            Number of partitions.

        Returns
        -------
        :class:`GroupedMatrixTable`
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
        """Aggregate by group, used after :meth:`MatrixTable.group_rows_by` or :meth:`MatrixTable.group_cols_by`.

        Examples
        --------
        Aggregate to a matrix with genes as row keys, computing the number of
        non-reference calls as an entry field:

        >>> dataset_result = (dataset.group_rows_by(dataset.gene)
        ...                          .aggregate(n_non_ref = agg.count_where(dataset.GT.is_non_ref())))

        Parameters
        ----------
        named_exprs : varargs of :class:`hail.expr.expression.Expression`
            Aggregation expressions.

        Returns
        -------
        :class:`.MatrixTable`
            Aggregated matrix table.
        """
        named_exprs = {k: to_expr(v) for k, v in named_exprs.items()}

        strs = []

        base, cleanup = self._parent._process_joins(*((self._group,) + tuple(named_exprs.values())))
        for k, v in named_exprs.items():
            analyze('GroupedMatrixTable.aggregate', v, self._grouped_indices,
                    {self._parent._row_axis, self._parent._col_axis})
            replace_aggregables(v._ast, 'gs')
            strs.append('`{}` = {}'.format(k, v._ast.to_hql()))

        if self._grouped_indices == self._parent._row_indices:
            # group variants
            return cleanup(
                MatrixTable(base._jvds.groupVariantsBy(self._group._ast.to_hql(), ',\n'.join(strs))))
        else:
            assert self._grouped_indices == self._parent._col_indices
            # group samples
            return cleanup(
                MatrixTable(base._jvds.groupSamplesBy(self._group._ast.to_hql(), ',\n'.join(strs))))

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
        self._sample_annotations = None
        self._colkey_schema = None
        self._sa_schema = None
        self._rowkey_schema = None
        self._va_schema = None
        self._global_schema = None
        self._genotype_schema = None
        self._sample_ids = None
        self._num_samples = None
        self._row_axis = 'row'
        self._col_axis = 'column'
        self._global_indices = Indices(self, set())
        self._row_indices = Indices(self, {self._row_axis})
        self._col_indices = Indices(self, {self._col_axis})
        self._entry_indices = Indices(self, {self._row_axis, self._col_axis})
        self._reserved = {'v', 's'}
        self._fields = {}

        assert isinstance(self.global_schema, TStruct), self.col_schema
        assert isinstance(self.col_schema, TStruct), self.col_schema
        assert isinstance(self.row_schema, TStruct), self.row_schema
        assert isinstance(self.entry_schema, TStruct), self.entry_schema

        self._set_field('v', construct_reference('v', self.rowkey_schema, self._row_indices))
        self._set_field('s', construct_reference('s', self.colkey_schema, self._col_indices))

        for f in self.global_schema.fields:
            assert f.name not in self._reserved, f.name
            self._set_field(f.name, construct_reference(f.name, f.typ, self._global_indices, prefix='global'))

        for f in self.col_schema.fields:
            assert f.name not in self._reserved, f.name
            self._set_field(f.name, construct_reference(f.name, f.typ, self._col_indices, prefix='sa'))

        for f in self.row_schema.fields:
            assert f.name not in self._reserved, f.name
            self._set_field(f.name, construct_reference(f.name, f.typ, self._row_indices, prefix='va'))

        for f in self.entry_schema.fields:
            assert f.name not in self._reserved, f.name
            self._set_field(f.name, construct_reference(f.name, f.typ, self._entry_indices, prefix='g'))

    def _set_field(self, key, value):
        assert key not in self._fields, key
        self._fields[key] = value
        if key in dir(self):
            warn("Name collision: field '{}' already in object dict."
                 " This field must be referenced with indexing syntax".format(key))
        else:
            self.__dict__[key] = value

    @typecheck_method(item=strlike)
    def _get_field(self, item):
        if item in self._fields:
            return self._fields[item]
        else:
            # no field detected
            raise KeyError("No field '{name}' found. "
                           "Global fields: [{global_fields}], "
                           "Row-indexed fields: [{row_fields}], "
                           "Column-indexed fields: [{col_fields}], "
                           "Row/Column-indexed fields: [{entry_fields}]".format(
                name=item,
                global_fields=', '.join(repr(f.name) for f in self.global_schema.fields),
                row_fields=', '.join(repr(f.name) for f in self.row_schema.fields),
                col_fields=', '.join(repr(f.name) for f in self.col_schema.fields),
                entry_fields=', '.join(repr(f.name) for f in self.entry_schema.fields),
            ))

    def __delattr__(self, item):
        if not item[0] == '_':
            raise NotImplementedError('Dataset objects are not mutable')

    def __setattr__(self, key, value):
        if not key[0] == '_':
            raise NotImplementedError('Dataset objects are not mutable')
        self.__dict__[key] = value

    def __getattr__(self, item):
        if item in self.__dict__:
            return self.__dict__[item]
        else:
            return self[item]

    @typecheck_method(item=oneof(strlike, sized_tupleof(oneof(slice, Expression), oneof(slice, Expression))))
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
                row_key = to_expr(exprs[0])
                if row_key._type != self.rowkey_schema:
                    raise ExpressionException(
                        'Type mismatch for MatrixTable row key: expected key type {}, found {}'.format(
                            str(self.rowkey_schema), str(row_key._type)))

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
                col_key = to_expr(exprs[1])
                if col_key._type != self.colkey_schema:
                    raise ExpressionException(
                        'Type mismatch for MatrixTable col key: expected key type {}, found {}'.format(
                            str(self.colkey_schema), str(col_key._type)))

            if row_key is not None and col_key is not None:
                return self.view_join_entries(row_key, col_key)
            elif row_key is not None and col_key is None:
                return self.view_join_rows(row_key)
            elif row_key is None and col_key is not None:
                return self.view_join_cols(col_key)
            else:
                return self.view_join_globals()

    @property
    def global_schema(self):
        """The schema of global fields in the matrix.

        Returns
        -------
        :class:`hail.expr.TStruct`
            Global schema.
        """
        if self._global_schema is None:
            self._global_schema = Type._from_java(self._jvds.globalSignature())
        return self._global_schema

    @property
    def colkey_schema(self):
        """The schema of the column key.

        Returns
        -------
        :class:`hail.expr.Type`
             Column key schema.
        """
        if self._colkey_schema is None:
            self._colkey_schema = Type._from_java(self._jvds.sSignature())
        return self._colkey_schema

    @property
    def col_schema(self):
        """The schema of column-indexed fields in the matrix.

        Returns
        -------
        :class:`hail.expr.TStruct`
             Column schema.
        """
        if self._sa_schema is None:
            self._sa_schema = Type._from_java(self._jvds.saSignature())
        return self._sa_schema

    @property
    def rowkey_schema(self):
        """The schema of the row key.

        Returns
        -------
        :class:`hail.expr.Type`
             Row key schema.
        """
        if self._rowkey_schema is None:
            self._rowkey_schema = Type._from_java(self._jvds.vSignature())
        return self._rowkey_schema

    @property
    def row_schema(self):
        """The schema of row-indexed fields in the matrix.

        Returns
        -------
        :class:`hail.expr.TStruct`
             Row schema.
        """
        if self._va_schema is None:
            self._va_schema = Type._from_java(self._jvds.vaSignature())
        return self._va_schema

    @property
    def entry_schema(self):
        """The schema of row-and-column-indexed fields in the matrix.

        Returns
        -------
        :class:`hail.expr.TStruct`
             Entry schema.
        """
        if self._genotype_schema is None:
            self._genotype_schema = Type._from_java(self._jvds.genotypeSignature())
        return self._genotype_schema

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

        The arguments to the method should either be :class:`hail.expr.expression.Expression`
        objects, or should be implicitly interpretable as expressions.

        Parameters
        ----------
        named_exprs : keyword args of :class:`hail.expr.expression.Expression`
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
            exprs.append('global.`{k}` = {v}'.format(k=k, v=v._ast.to_hql()))
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
        >>> dataset_result = dataset.annotate_rows(call_stats = agg.call_stats(high_quality_calls, dataset.v))

        Add functional annotations from a :class:`.Table` keyed by :class:`hail.expr.TVariant`:, and another
        :class:`.MatrixTable`.

        >>> dataset_result = dataset.annotate_rows(consequence = table1[dataset.v].consequence,
        ...                                        dataset2_AF = dataset2[dataset.v, :].info.AF)

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

        The arguments to the method should either be :class:`hail.expr.expression.Expression`
        objects, or should be implicitly interpretable as expressions.

        Parameters
        ----------
        named_exprs : keyword args of :class:`hail.expr.expression.Expression`
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
            exprs.append('va.`{k}` = {v}'.format(k=k, v=v._ast.to_hql()))
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

        Add sample metadata from a :class:`hail.api2.Table`.

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

        The arguments to the method should either be :class:`hail.expr.expression.Expression`
        objects, or should be implicitly interpretable as expressions.

        Parameters
        ----------
        named_exprs : keyword args of :class:`hail.expr.expression.Expression`
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
            exprs.append('sa.`{k}` = {v}'.format(k=k, v=v._ast.to_hql()))
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

        The arguments to the method should either be :class:`hail.expr.expression.Expression`
        objects, or should be implicitly interpretable as expressions.

        Parameters
        ----------
        named_exprs : keyword args of :class:`hail.expr.expression.Expression`
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
            exprs.append('g.`{k}` = {v}'.format(k=k, v=v._ast.to_hql()))
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

        See :py:meth:`Table.select` for more information about using ``select`` methods.

        Note
        ----
        This method does not support aggregation.

        Parameters
        ----------
        exprs : variable-length args of :obj:`str` or :class:`hail.expr.expression.Expression`
            Arguments that specify field names or nested field reference expressions.
        named_exprs : keyword args of :class:`hail.expr.expression.Expression`
            Field names and the expressions to compute them.

        Returns
        -------
        :class:`.MatrixTable`
            MatrixTable with specified global fields.
        """
        exprs = tuple(to_expr(e) if not isinstance(e, str) and not isinstance(e, unicode) else self[e] for e in exprs)
        named_exprs = {k: to_expr(v) for k, v in named_exprs.items()}
        strs = []
        all_exprs = []
        base, cleanup = self._process_joins(*(exprs + tuple(named_exprs.values())))
        for e in exprs:
            all_exprs.append(e)
            analyze('MatrixTable.select_globals', e, self._global_indices)
            if e._ast.search(lambda ast: not isinstance(ast, Reference) and not isinstance(ast, Select)):
                raise ExpressionException("method 'select_globals' expects keyword arguments for complex expressions")
            strs.append(
                '`{}`: {}'.format(e._ast.selection if isinstance(e._ast, Select) else e._ast.name, e._ast.to_hql()))
        for k, e in named_exprs.items():
            all_exprs.append(e)
            analyze('MatrixTable.select_globals', e, self._global_indices)
            self._check_field_name(k, self._global_indices)
            strs.append('`{}`: {}'.format(k, to_expr(e)._ast.to_hql()))
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

        See :py:meth:`Table.select` for more information about using ``select`` methods.

        Note
        ----
        This method supports aggregation over columns. For instance, the usage:

        >>> dataset_result = dataset.select_rows(mean_GQ = agg.mean(dataset.GQ))

        will compute the mean per row.

        Parameters
        ----------
        exprs : variable-length args of :obj:`str` or :class:`hail.expr.expression.Expression`
            Arguments that specify field names or nested field reference expressions.
        named_exprs : keyword args of :class:`hail.expr.expression.Expression`
            Field names and the expressions to compute them.

        Returns
        -------
        :class:`.MatrixTable`
            MatrixTable with specified row fields.
        """
        exprs = tuple(to_expr(e) if not isinstance(e, str) and not isinstance(e, unicode) else self[e] for e in exprs)
        named_exprs = {k: to_expr(v) for k, v in named_exprs.items()}
        strs = []
        all_exprs = []
        base, cleanup = self._process_joins(*(exprs + tuple(named_exprs.values())))

        for e in exprs:
            all_exprs.append(e)
            analyze('MatrixTable.select_rows', e, self._row_indices, {self._col_axis})
            if e._ast.search(lambda ast: not isinstance(ast, Reference) and not isinstance(ast, Select)):
                raise ExpressionException("method 'select_rows' expects keyword arguments for complex expressions")
            replace_aggregables(e._ast, 'gs')
            strs.append('`{}`: {}'.format(e._ast.selection if isinstance(e._ast, Select) else e._ast.name,
                                          e._ast.to_hql()))
        for k, e in named_exprs.items():
            all_exprs.append(e)
            analyze('MatrixTable.select_rows', e, self._row_indices, {self._col_axis})
            self._check_field_name(k, self._row_indices)
            replace_aggregables(e._ast, 'gs')
            strs.append('`{}`: {}'.format(k, e._ast.to_hql()))
        m = MatrixTable(base._jvds.annotateVariantsExpr('va = {' + ',\n'.join(strs) + '}'))
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

        See :py:meth:`Table.select` for more information about using ``select`` methods.

        Note
        ----
        This method supports aggregation over rows. For instance, the usage:

        >>> dataset_result = dataset.select_cols(mean_GQ = agg.mean(dataset.GQ))

        will compute the mean per column.

        Parameters
        ----------
        exprs : variable-length args of :obj:`str` or :class:`hail.expr.expression.Expression`
            Arguments that specify field names or nested field reference expressions.
        named_exprs : keyword args of :class:`hail.expr.expression.Expression`
            Field names and the expressions to compute them.

        Returns
        -------
        :class:`.MatrixTable`
            MatrixTable with specified column fields.
        """

        exprs = tuple(to_expr(e) if not isinstance(e, str) and not isinstance(e, unicode) else self[e] for e in exprs)
        named_exprs = {k: to_expr(v) for k, v in named_exprs.items()}
        strs = []
        all_exprs = []
        base, cleanup = self._process_joins(*(exprs + tuple(named_exprs.values())))

        for e in exprs:
            all_exprs.append(e)
            analyze('MatrixTable.select_cols', e, self._col_indices, {self._row_axis})
            if e._ast.search(lambda ast: not isinstance(ast, Reference) and not isinstance(ast, Select)):
                raise ExpressionException("method 'select_cols' expects keyword arguments for complex expressions")
            replace_aggregables(e._ast, 'gs')
            strs.append('`{}`: {}'.format(e._ast.selection if isinstance(e._ast, Select) else e._ast.name,
                                          e._ast.to_hql()))
        for k, e in named_exprs.items():
            all_exprs.append(e)
            analyze('MatrixTable.select_cols', e, self._col_indices, {self._row_axis})
            self._check_field_name(k, self._col_indices)
            replace_aggregables(e._ast, 'gs')
            strs.append('`{}`: {}'.format(k, e._ast.to_hql()))

        m = MatrixTable(base._jvds.annotateSamplesExpr('sa = {' + ',\n'.join(strs) + '}'))
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

        See :py:meth:`Table.select` for more information about using ``select`` methods.

        Note
        ----
        This method does not support aggregation.

        Parameters
        ----------
        exprs : variable-length args of :obj:`str` or :class:`hail.expr.expression.Expression`
            Arguments that specify field names or nested field reference expressions.
        named_exprs : keyword args of :class:`hail.expr.expression.Expression`
            Field names and the expressions to compute them.

        Returns
        -------
        :class:`.MatrixTable`
            MatrixTable with specified entry fields.
        """
        exprs = tuple(to_expr(e) if not isinstance(e, str) and not isinstance(e, unicode) else self[e] for e in exprs)
        named_exprs = {k: to_expr(v) for k, v in named_exprs.items()}
        strs = []
        all_exprs = []
        base, cleanup = self._process_joins(*(exprs + tuple(named_exprs.values())))

        for e in exprs:
            all_exprs.append(e)
            analyze('MatrixTable.select_entries', e, self._entry_indices)
            if e._ast.search(lambda ast: not isinstance(ast, Reference) and not isinstance(ast, Select)):
                raise ExpressionException("method 'select_globals' expects keyword arguments for complex expressions")
            strs.append(
                '`{}`: {}'.format(e._ast.selection if isinstance(e._ast, Select) else e._ast.name, e._ast.to_hql()))
        for k, e in named_exprs.items():
            all_exprs.append(e)
            analyze('MatrixTable.select_entries', e, self._entry_indices)
            self._check_field_name(k, self._entry_indices)
            strs.append('`{}`: {}'.format(k, e._ast.to_hql()))
        m = MatrixTable(base._jvds.annotateGenotypesExpr('g = {' + ',\n'.join(strs) + '}'))
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
        exprs : varargs of :obj:`str` or :class:`hail.expr.expression.Expression`
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
            new_global_fields = {k.name: m._fields[k.name] for k in m.global_schema.fields if
                                 k.name not in fields_to_drop}
            m = m.select_globals(**new_global_fields)

        if any(self._fields[field]._indices == self._row_indices for field in fields_to_drop):
            # need to drop row fields
            new_row_fields = {k.name: m._fields[k.name] for k in m.row_schema.fields if k.name not in fields_to_drop}
            m = m.select_rows(**new_row_fields)

        if any(self._fields[field]._indices == self._col_indices for field in fields_to_drop):
            # need to drop col fields
            new_col_fields = {k.name: m._fields[k.name] for k in m.col_schema.fields if k.name not in fields_to_drop}
            m = m.select_cols(**new_col_fields)

        if any(self._fields[field]._indices == self._entry_indices for field in fields_to_drop):
            # need to drop entry fields
            new_entry_fields = {k.name: m._fields[k.name] for k in m.entry_schema.fields if
                                k.name not in fields_to_drop}
            m = m.select_entries(**new_entry_fields)

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
        return MatrixTable(self._jvds.dropVariants())

    def drop_cols(self):
        """Drop all columns of the matrix.  Is equivalent to:

        >>> dataset_result = dataset.filter_cols(False)

        Returns
        -------
        :class:`.MatrixTable`
            Matrix table with no columns.
        """
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
        expr : bool or :class:`hail.expr.expression.BooleanExpression`
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
        expr : bool or :class:`hail.expr.expression.BooleanExpression`
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
        expr : bool or :class:`hail.expr.expression.BooleanExpression`
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
        """Similar to :meth:`MatrixTable.annotate_globals`, but drops referenced fields.

        Note
        ----
        Not implemented.

        Parameters
        ----------
        named_exprs : keyword args of :class:`hail.expr.expression.Expression`
            Annotation expressions.

        Returns
        -------
        :class:`.MatrixTable`
            Annotated matrix table.
        """
        raise NotImplementedError()

    def transmute_rows(self, **named_exprs):
        """Similar to :meth:`MatrixTable.annotate_rows`, but drops referenced fields.

        Note
        ----
        Not implemented.

        Parameters
        ----------
        named_exprs : keyword args of :class:`hail.expr.expression.Expression`
            Annotation expressions.

        Returns
        -------
        :class:`.MatrixTable`
            Annotated matrix table.
        """

        raise NotImplementedError()

    @handle_py4j
    def transmute_cols(self, **named_exprs):
        """Similar to :meth:`MatrixTable.annotate_cols`, but drops referenced fields.

        Note
        ----
        Not implemented.

        Parameters
        ----------
        named_exprs : keyword args of :class:`hail.expr.expression.Expression`
            Annotation expressions.

        Returns
        -------
        :class:`.MatrixTable`
            Annotated matrix table.
        """
        raise NotImplementedError()

    @handle_py4j
    def transmute_entries(self, **named_exprs):
        """Similar to :meth:`MatrixTable.annotate_entries`, but drops referenced fields.

        Note
        ----
        Not implemented.

        Parameters
        ----------
        named_exprs : keyword args of :class:`hail.expr.expression.Expression`
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
        named_exprs : keyword args of :class:`hail.expr.expression.Expression`
            Aggregation expressions.

        Returns
        -------
        :class:`Struct`
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
        named_exprs : keyword args of :class:`hail.expr.expression.Expression`
            Aggregation expressions.

        Returns
        -------
        :class:`Struct`
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
        named_exprs : keyword args of :class:`hail.expr.expression.Expression`
            Aggregation expressions.

        Returns
        -------
        :class:`Struct`
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
        field_expr : str or :class:`hail.expr.expression.Expression`
            Field name or (possibly nested) field reference expression.

        Returns
        -------
        :class:MatrixTable`
            Matrix table exploded row-wise for each element of `field_expr`.
        """
        if isinstance(field_expr, str) or isinstance(field_expr, unicode):
            if not field_expr in self._fields:
                raise KeyError("MatrixTable has no field '{}'".format(field_expr))
            elif self._fields[field_expr].indices != self._row_indices:
                raise ExpressionException("Method 'explode_rows' expects a field indexed by row, found axes '{}'"
                                          .format(self._fields[field_expr].indices.axes))
            s = 'va.`{}`'.format(field_expr)
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
        field_expr : str or :class:`hail.expr.expression.Expression`
            Field name or (possibly nested) field reference expression.

        Returns
        -------
        :class:MatrixTable`
            Matrix table exploded column-wise for each element of `field_expr`.
        """

        if isinstance(field_expr, str) or isinstance(field_expr, unicode):
            if not field_expr in self._fields:
                raise KeyError("MatrixTable has no field '{}'".format(field_expr))
            elif self._fields[field_expr].indices != self._col_indices:
                raise ExpressionException("Method 'explode_cols' expects a field indexed by col, found axes '{}'"
                                          .format(self._fields[field_expr].indices.axes))
            s = 'sa.`{}`'.format(field_expr)
        else:
            e = to_expr(field_expr)
            analyze('MatrixTable.explode_cols', field_expr, self._col_indices)
            if e._ast.search(lambda ast: not isinstance(ast, Reference) and not isinstance(ast, Select)):
                raise ExpressionException(
                    "method 'explode_cols' requires a field or subfield, not a complex expression")
            s = e._ast.to_hql()
        return MatrixTable(self._jvds.explodeSamples(s))

    @handle_py4j
    def group_rows_by(self, key_expr):
        """Group rows, used with :meth:`GroupedMatrixTable.aggregate`

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
        key_expr : str or :class:`hail.expr.expression.Expression`
            Field name or expression to use as new row key.

        Returns
        -------
        :class:`GroupedMatrixTable`
            Grouped matrix, can be used to call :meth:`GroupedMatrixTable.aggregate`.
        """
        if isinstance(key_expr, str) or isinstance(key_expr, unicode):
            key_expr = self[key_expr]
        key_expr = to_expr(key_expr)
        analyze('MatrixTable.group_rows_by', key_expr, self._row_indices, {self._col_axis})
        return GroupedMatrixTable(self, key_expr, self._row_indices)

    @handle_py4j
    def group_cols_by(self, key_expr):
        """Group rows, used with :meth:`GroupedMatrixTable.aggregate`

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
        key_expr : str or :class:`hail.expr.expression.Expression`
            Field name or expression to use as new column key.

        Returns
        -------
        :class:`GroupedMatrixTable`
            Grouped matrix, can be used to call :meth:`GroupedMatrixTable.aggregate`.
        """
        if isinstance(key_expr, str) or isinstance(key_expr, unicode):
            key_expr = self[key_expr]
        key_expr = to_expr(key_expr)
        analyze('MatrixTable.group_cols_by', key_expr, self._col_indices, {self._row_axis})
        return GroupedMatrixTable(self, key_expr, self._col_indices)

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
        return self._jvds.nSamples()

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
        kt = Table(self._jvds.variantsKT())

        # explode the 'va' struct to the top level
        return kt.select(kt.v, *kt.va)

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
        kt = Table(self._jvds.samplesKT())

        # explode the 'sa' struct to the top level
        return kt.select(kt.s, *kt.sa)

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
        kt = Table(self._jvds.genotypeKT())

        # explode the 'va', 'sa', 'g' structs to the top level
        # FIXME: this part should really be in Scala
        cols_to_select = tuple(x for x in kt.va) + tuple(x for x in kt.sa) + tuple(x for x in kt.g)
        return kt.select(kt.v, kt.s, *cols_to_select)

    @handle_py4j
    def view_join_globals(self):
        uid = Env._get_uid()

        def joiner(obj):
            if isinstance(obj, MatrixTable):
                return MatrixTable(Env.jutils().joinGlobals(obj._jvds, self._jvds, uid))
            else:
                assert isinstance(obj, Table)
                return Table(Env.jutils().joinGlobals(obj._jt, self._jvds, uid))

        return construct_expr(GlobalJoinReference(uid), self.global_schema, joins=LinkedList(Join).push(Join(joiner, [uid])))

    @handle_py4j
    def view_join_rows(self, expr):
        expr = to_expr(expr)
        indices, aggregations, joins, refs = expr._indices, expr._aggregations, expr._joins, expr._refs
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
            select_struct = Struct(**{k: right[k] for k in [f.name for f in self.row_schema.fields]})
            right = right.select(right.v, **{uid: select_struct})

            key_uid = Env._get_uid()
            uids_to_delete.append(key_uid)

            def joiner(left):
                pre_key = left.key
                left = Table(left._jt.annotate('{} = {}'.format(key_uid, expr._ast.to_hql())))
                left = left.key_by(key_uid)
                left = left.to_hail1().join(right.to_hail1(), 'left').to_hail2()
                left = left.key_by(*pre_key)
                return left

            return construct_expr(Reference(uid), self.row_schema, indices, aggregations,
                                  joins.push(Join(joiner, uids_to_delete)))
        else:
            assert isinstance(src, MatrixTable)
            right = self

            # fast path
            if expr is src.v:
                prefix = 'va'
                joiner = lambda left: (
                    MatrixTable(left._jvds.annotateVariantsVDS(right._jvds, jsome('{}.{}'.format(prefix, uid)),
                                                               jnone())))
            elif indices == src._row_indices:
                prefix = 'va'
                joiner = lambda left: (
                    MatrixTable(left._jvds.annotateVariantsTable(right._jvds.variantsKT(),
                                                                 [expr._ast.to_hql()],
                                                                 '{}.{}'.format(prefix, uid), None, False)))
            elif indices == src._col_indices:
                prefix = 'sa'
                joiner = lambda left: (
                    MatrixTable(left._jvds.annotateSamplesTable(right._jvds.variantsKT(),
                                                                [expr._ast.to_hql()],
                                                                '{}.{}'.format(prefix, uid), None, False)))
            else:
                # FIXME: implement entry-based join in the expression language
                raise NotImplementedError('MatrixTable.view_join_rows with indices {}'.format(indices))

            return construct_expr(Select(Reference(prefix), uid),
                                  self.row_schema, indices, aggregations,
                                  joins.push(Join(joiner, uids_to_delete)), refs)

    @handle_py4j
    def view_join_cols(self, expr):
        expr = to_expr(expr)
        indices, aggregations, joins, refs = expr._indices, expr._aggregations, expr._joins, expr._refs
        src = indices.source

        if aggregations:
            raise ExpressionException('Cannot join using an aggregated field')
        uid = Env._get_uid()
        uids_to_delete = [uid]

        if src is None:
            raise ExpressionException('Cannot index with a scalar expression')

        if isinstance(src, Table):
            # join table with matrix.cols_table()
            right = self.cols_table()
            select_struct = Struct(**{k: right[k] for k in [f.name for f in self.col_schema.fields]})
            right = right.select(right.s, **{uid: select_struct})

            key_uid = Env._get_uid()
            uids_to_delete.append(key_uid)

            def joiner(left):
                pre_key = left.key
                left = Table(left._jt.annotate('{} = {}'.format(key_uid, expr._ast.to_hql())))
                left = left.key_by(key_uid)
                left = left.to_hail1().join(right.to_hail1(), 'left').to_hail2()
                left = left.key_by(*pre_key)
                return left

            return construct_expr(Reference(uid),
                                  self.col_schema, indices, aggregations, joins.push(Join(joiner, uids_to_delete)))
        else:
            assert isinstance(src, MatrixTable)
            right = self

            if indices == src._col_indices:
                prefix = 'sa'
                joiner = lambda left: (
                    MatrixTable(left._jvds.annotateSamplesTable(right._jvds.samplesKT(),
                                                                [expr._ast.to_hql()],
                                                                '{}.{}'.format(prefix, uid), None, False)))
            elif indices == src._row_indices:
                prefix = 'va'
                joiner = lambda left: (
                    MatrixTable(left._jvds.annotateVariantsTable(right._jvds.samplesKT(),
                                                                 [expr._ast.to_hql()],
                                                                 '{}.{}'.format(prefix, uid), None, False)))
            else:
                # FIXME: implement entry-based join in the expression language
                raise NotImplementedError('MatrixTable.view_join_cols join with indices {}'.format(indices))
            return construct_expr(Select(Reference(prefix), uid),
                                  self.col_schema, indices, aggregations,
                                  joins.push(Join(joiner, uids_to_delete)), refs)

    @handle_py4j
    def view_join_entries(self, row_expr, col_expr):
        row_expr = to_expr(row_expr)
        col_expr = to_expr(col_expr)

        indices, aggregations, joins, refs = unify_all(row_expr, col_expr)
        src = indices.source
        if aggregations:
            raise ExpressionException('Cannot join using an aggregated field')
        uid = Env._get_uid()
        uids_to_delete = [uid]

        if isinstance(src, Table):
            # join table with matrix.entries_table()
            right = self.entries_table()
            select_struct = Struct(**{k: right[k] for k in [f.name for f in self.entry_schema.fields]})
            right = right.select(right.v, right.s, **{uid: select_struct})

            row_key_uid = Env._get_uid()
            col_key_uid = Env._get_uid()
            uids_to_delete.append(row_key_uid)
            uids_to_delete.append(col_key_uid)

            def joiner(left):
                pre_key = left.key
                left = Table(left._jt.annotate('{} = {}, {} = {}'.format(
                    row_key_uid, row_expr._ast.to_hql(),
                    col_key_uid, col_expr._ast.to_hql())))
                left = left.key_by(row_key_uid, col_key_uid)
                left = left.to_hail1().join(right.to_hail1(), 'left').to_hail2()
                left = left.key_by(*pre_key)
                return left

            return construct_expr(Reference(uid),
                                  self.entry_schema, indices, aggregations,
                                  joins.push(Join(joiner, uids_to_delete)), refs)
        else:
            raise NotImplementedError('matrix.view_join_entries with {}'.format(src.__class__))

    def to_hail1(self):
        """Convert to a hail1 variant dataset.

        Returns
        -------
        :class:`hail.api1.VariantDataset`
        """
        import hail
        h1vds = hail.VariantDataset(Env.hc(), self._jvds)
        h1vds._set_history(History('is a mystery'))
        return h1vds

    @typecheck_method(name=strlike, indices=Indices)
    def _check_field_name(self, name, indices):
        if name in self._reserved:
            msg = 'name collision with reserved namespace: {}'.format(name)
            error('Analysis exception: {}'.format(msg))
            raise ExpressionException(msg)
        if name in set(self._fields.keys()) and not self._fields[name]._indices == indices:
            msg = 'name collision with field indexed by {}: {}'.format(indices, name)
            error('Analysis exception: {}'.format(msg))
            raise ExpressionException(msg)

    @typecheck_method(exprs=Expression)
    def _process_joins(self, *exprs):

        all_uids = []
        left = self

        for e in exprs:
            rewrite_global_refs(e._ast, self)
            for j in list(e._joins)[::-1]:
                left = j.join_function(left)
                all_uids.extend(j.temp_vars)

        def cleanup(matrix):
            return matrix.drop(*all_uids)

        return left, cleanup

    @typecheck_method(truncate_at=integral)
    def describe(self, truncate_at=60):
        """Print information about the fields in the matrix."""
        def format_type(typ):
            typ_str = str(typ)
            if len(typ_str) > truncate_at - 3:
                typ_str = typ_str[:truncate_at - 3] + '...'
            return typ_str

        if len(self.global_schema.fields) == 0:
            global_fields = '\n    None'
        else:
            global_fields = ''.join("\n    '{name}': {type} ".format(
                name=fd.name, type=format_type(fd.typ)) for fd in self.global_schema.fields)

        row_fields = "\n    'v' [row key]: {}".format(format_type(self.rowkey_schema))
        row_fields += ''.join("\n    '{name}': {type} ".format(
            name=fd.name, type=format_type(fd.typ)) for fd in self.row_schema.fields)

        col_fields = "\n    's' [col key]: {}".format(format_type(self.colkey_schema))
        col_fields += ''.join("\n    '{name}': {type} ".format(
            name=fd.name, type=format_type(fd.typ)) for fd in self.col_schema.fields)

        if len(self.entry_schema.fields) == 0:
            entry_fields = '\n    None'
        else:
            entry_fields = ''.join("\n    '{name}': {type} ".format(
                name=fd.name, type=format_type(fd.typ)) for fd in self.entry_schema.fields)

        s = 'Global fields:{}\n\nRow-indexed fields:{}\n\nColumn-indexed fields:{}\n\nEntry-indexed fields:{}'.format(
            global_fields, row_fields, col_fields, entry_fields)
        print(s)

    @handle_py4j
    @typecheck_method(keys=list, keep=bool)
    def filter_rows_list(self, keys, keep=True):
        """Filter rows with a list of keys to keep or exclude.

        Examples
        --------
        Subset to two specific rows:

        >>> to_keep = [Variant.parse('20:10626633:G:GC'), Variant.parse('20:10019093:A:G')]
        >>> dataset_result = dataset.filter_rows_list(to_keep, keep=True)

        Notes
        -----

        This method is extremely efficient when `keep` is ``True``: data shards
        that don't overlap with any supplied key will not be loaded at all. This
        method can therefore be used for reasonably low-latency queries of one
        or more rows, even on large datasets.

        Parameters
        ----------
        keys : list
            List of row keys to keep or remove.
        keep : bool
            If ``True``, subset rows to those appearing in `keys`. If
            ``False``, remove rows in `keys`.

        Returns
        -------
        :class:`.MatrixTable`
            Matrix table with rows removed.
        """
        rk_type = self.rowkey_schema
        jkeys = []
        for k in keys:
            rk_type._typecheck(k)
            jkeys.append(rk_type._convert_to_j(k))

        return MatrixTable(
            self._jvds.filterVariantsList(jkeys, keep))

    @handle_py4j
    @typecheck_method(keys=list, keep=bool)
    def filter_cols_list(self, keys, keep=True):
        """Filter columns with a list of keys to keep or exclude.

        Examples
        --------
        Remove three columns:

        >>> to_remove = ['NA12878', 'NA12891', 'NA12892']
        >>> dataset_result = dataset.filter_cols_list(to_remove, keep=False)

        Parameters
        ----------
        keys : list
            List of column keys to keep or remove.
        keep : bool
            If ``True``, subset columns to those appearing in `keys`. If
            ``False``, remove columns in `keys`.

        Returns
        -------
        :class:`.MatrixTable`
            Matrix table with columns removed.
        """

        ck_type = self.colkey_schema
        jkeys = []
        for k in keys:
            ck_type._typecheck(k)
            jkeys.append(ck_type._convert_to_j(k))
        return MatrixTable(self._jvds.filterSamplesList(jkeys, keep))

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

        Check the current number of partitions with :meth:`num_partitions`.

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
        """Naively descrease the number of partitions.

        Example
        -------
        Naively repartition to 10 partitions:

        >>> dataset_result = dataset.naive_coalesce(10)

        Warning
        -------
        :meth:`naive_coalesce` simply combines adjacent partitions to achieve
        the desired number. It does not attempt to rebalance, unlike
        :meth:`repartition`, so it can produce a heavily unbalanced dataset. An
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

        This method is an alias for :func:`persist("MEMORY_ONLY") <hail.api2.MatrixTable.persist>`.

        Returns
        -------
        :class:`.MatrixTable`
            Cached dataset.
        """
        return self.persist('MEMORY_ONLY')

    @typecheck_method(storage_level=enumeration('NONE', 'DISK_ONLY', 'DISK_ONLY_2', 'MEMORY_ONLY',
                                                'MEMORY_ONLY_2', 'MEMORY_ONLY_SER', 'MEMORY_ONLY_SER_2',
                                                'MEMORY_AND_DISK', 'MEMORY_AND_DISK_2', 'MEMORY_AND_DISK_SER',
                                                'MEMORY_AND_DISK_SER_2', 'OFF_HEAP'))
    def persist(self, storage_level='MEMORY_AND_DISK'):
        """Persist this table in memory or on disk.

        Examples
        --------
        Persist the dataset to both memory and disk:

        >>> dataset = dataset.persist() # doctest: +SKIP

        Notes
        -----

        The :py:meth:`MatrixTable.persist` and :py:meth:`MatrixTable.cache`
        methods store the current dataset on disk or in memory temporarily to
        avoid redundant computation and improve the performance of Hail
        pipelines. This method is not a substitution for :py:meth:`Table.write`,
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

matrix_table_type.set(MatrixTable)
