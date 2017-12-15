from __future__ import print_function  # Python 2 and 3 print compatibility

import hail.expr.functions as f

from hail.expr.expression import *
from hail.utils import wrap_to_list


class TableTemplate(HistoryMixin):
    def __init__(self, hc, jkt):
        self._hc = hc
        self._jkt = jkt

        self._globals = None
        self._global_schema = None
        self._schema = None
        self._num_columns = None
        self._key = None
        self._column_names = None
        self._fields = {}

    def _set_field(self, key, value):
        self._fields[key] = value
        if key in dir(self):
            warn("Name collision: field '{}' already in object dict. "
                 "This field must be referenced with indexing syntax".format(key))
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
                           "Row-indexed fields: [{row_fields}]".format(
                name=item,
                global_fields=', '.join(repr(f.name) for f in self.global_schema.fields),
                row_fields=', '.join(repr(f.name) for f in self.schema.fields),
            ))

    def __getitem__(self, item):
        return self._get_field(item)

    def __delattr__(self, item):
        if not item[0] == '_':
            raise NotImplementedError('Table objects are not mutable')

    def __setattr__(self, key, value):
        if not key[0] == '_':
            raise NotImplementedError('Table objects are not mutable')
        self.__dict__[key] = value

    def __getattr__(self, item):
        if item in self.__dict__:
            return self.__dict__[item]
        else:
            return self[item]

    def __repr__(self):
        return self._jkt.toString()

    @handle_py4j
    def globals(self):
        if self._globals is None:
            self._globals = self.global_schema._convert_to_py(self._jkt.globals())
        return self._globals

    @property
    @handle_py4j
    def schema(self):
        if self._schema is None:
            self._schema = Type._from_java(self._jkt.signature())
            assert (isinstance(self._schema, TStruct))
        return self._schema

    @property
    @handle_py4j
    def global_schema(self):
        if self._global_schema is None:
            self._global_schema = Type._from_java(self._jkt.globalSignature())
            assert (isinstance(self._global_schema, TStruct))
        return self._global_schema

    @property
    @handle_py4j
    def key(self):
        if self._key is None:
            self._key = list(self._jkt.key())
        return self._key


class GroupedTable(TableTemplate):
    """Table that has been grouped.

    There are only two operations on a grouped table, :meth:`GroupedTable.partition_hint`
    and :meth:`GroupedTable.aggregate`.

    .. testsetup ::

        table1 = hc.import_table('data/kt_example1.tsv', impute=True, key='ID').to_hail2()
        from hail2 import *

    """

    def __init__(self, parent, groups):
        super(GroupedTable, self).__init__(parent._hc, parent._jkt)
        self._groups = groups
        self._parent = parent
        self._npartitions = None

        for fd in parent._fields:
            self._set_field(fd, parent._fields[fd])

    @handle_py4j
    @typecheck_method(n=integral)
    def partition_hint(self, n):
        """Set the target number of partitions for aggregation.

        Examples
        --------

        Use `partition_hint` in a :meth:`Table.group_by` / :meth:`GroupedTable.aggregate`
        pipeline:

        >>> table_result = (table1.group_by(table1.ID)
        ...                       .partition_hint(5)
        ...                       .aggregate(meanX = f.mean(table1.X), sumZ = f.sum(table1.Z)))

        Notes
        -----
        Until Hail's query optimizer is intelligent enough to sample records at all
        stages of a pipeline, it can be necessary in some places to provide some
        explicit hints.

        The default number of partitions for :meth:`GroupedTable.aggregate` is the
        number of partitions in the upstream table. If the aggregation greatly
        reduces the size of the table, providing a hint for the target number of
        partitions can accelerate downstream operations.

        Parameters
        ----------
        n : int
            Number of partitions.

        Returns
        -------
        :class:`GroupedTable`
            Same grouped table with a partition hint.
        """
        self._npartitions = n
        return self

    @handle_py4j
    @typecheck_method(named_exprs=dictof(strlike, anytype))
    def aggregate(self, **named_exprs):
        """Aggregate by group, used after :meth:`Table.group_by`.

        Examples
        --------
        Compute the mean value of `X` and the sum of `Z` per unique `ID`:

        >>> table_result = (table1.group_by(table1.ID)
        ...                       .aggregate(meanX = f.mean(table1.X), sumZ = f.sum(table1.Z)))

        Group by a height bin and compute sex ratio per bin:

        >>> table_result = (table1.group_by(height_bin = (table1.HT / 20).to_int32())
        ...                       .aggregate(fraction_female = f.fraction(table1.SEX == 'F')))

        Parameters
        ----------
        named_exprs : varargs of :class:`hail.expr.expression.Expression`
            Aggregation expressions.

        Returns
        -------
        :class:`Table`
            Aggregated table.
        """
        agg_base = self._parent.columns[0]  # FIXME hack

        named_exprs = {k: to_expr(v) for k, v in named_exprs.items()}

        strs = []
        base, cleanup = self._parent._process_joins(*(tuple(v for _, v in self._groups) + tuple(named_exprs.values())))
        for k, v in named_exprs.items():
            analyze(v, self._parent._global_indices, {self._parent._row_axis}, set(self._parent.columns))
            replace_aggregables(v._ast, agg_base)
            strs.append('`{}` = {}'.format(k, v._ast.to_hql()))

        group_strs = ',\n'.join('`{}` = {}'.format(k, v._ast.to_hql()) for k, v in self._groups)
        return cleanup(
            Table(self._hc, base._jkt.aggregate(group_strs, ",\n".join(strs), joption(self._npartitions))))


class Table(TableTemplate):
    """Hail's distributed implementation of a dataframe or SQL table.

    In the examples below, we have imported two key tables from text files (``table1`` and ``table2``).

    .. testsetup ::

        hc.stop()
        from hail2 import *
        hc = HailContext()

    >>> table1 = hc.import_table('data/kt_example1.tsv', impute=True, key='ID')
    >>> table1.show()

    .. code-block:: text

        +-------+-------+--------+-------+-------+-------+-------+-------+
        |    ID |    HT | SEX    |     X |     Z |    C1 |    C2 |    C3 |
        +-------+-------+--------+-------+-------+-------+-------+-------+
        | Int32 | Int32 | String | Int32 | Int32 | Int32 | Int32 | Int32 |
        +-------+-------+--------+-------+-------+-------+-------+-------+
        |     1 |    65 | M      |     5 |     4 |     2 |    50 |     5 |
        |     2 |    72 | M      |     6 |     3 |     2 |    61 |     1 |
        |     3 |    70 | F      |     7 |     3 |    10 |    81 |    -5 |
        |     4 |    60 | F      |     8 |     2 |    11 |    90 |   -10 |
        +-------+-------+--------+-------+-------+-------+-------+-------+

    >>> table2 = hc.import_table('data/kt_example2.tsv', impute=True, key='ID')
    >>> table2.show()

    .. code-block:: text

        +-------+-------+--------+
        |    ID |     A | B      |
        +-------+-------+--------+
        | Int32 | Int32 | String |
        +-------+-------+--------+
        |     1 |    65 | cat    |
        |     2 |    72 | dog    |
        |     3 |    70 | mouse  |
        |     4 |    60 | rabbit |
        +-------+-------+--------+

    Define new annotations:

    >>> height_mean_m = 68
    >>> height_sd_m = 3
    >>> height_mean_f = 65
    >>> height_sd_f = 2.5
    >>>
    >>> def get_z(height, sex):
    ...    return f.cond(sex == 'M',
    ...                  (height - height_mean_m) / height_sd_m,
    ...                  (height - height_mean_f) / height_sd_f)
    >>>
    >>> table1 = table1.annotate(height_z = get_z(table1.HT, table1.SEX))
    >>> table1 = table1.annotate_globals(global_field_1 = [1, 2, 3])

    Filter rows of the table:

    >>> table2 = table2.filter(table2.B != 'rabbit')

    Compute global aggregation statistics:

    >>> t1_stats = table1.aggregate(mean_c1 = f.mean(table1.C1),
    ...                             mean_c2 = f.mean(table1.C2),
    ...                             stats_c3 = f.stats(table1.C3))
    >>> print(t1_stats)

    Group columns and aggregate to produce a new table:

    >>> table3 = (table1.group_by(table1.SEX)
    ...                 .aggregate(mean_height_data = f.mean(table1.HT)))
    >>> table3.show()

    Join tables together inside an annotation expression:

    >>> table2 = table2.key_by('ID')
    >>> table1 = table1.annotate(B = table2[table1.ID].B)
    >>> table1.show()
    """

    def __init__(self, hc, jkt):
        super(Table, self).__init__(hc, jkt)
        self._global_indices = Indices(axes=set(), source=self)
        self._row_axis = 'row'
        self._row_indices = Indices(axes={self._row_axis}, source=self)

        for fd in self.global_schema.fields:
            column = construct_expr(Reference(fd.name), fd.typ, indices=self._global_indices, aggregations=(), joins=())
            self._set_field(fd.name, column)

        for fd in self.schema.fields:
            column = construct_expr(Reference(fd.name), fd.typ, indices=self._row_indices, aggregations=(), joins=())
            self._set_field(fd.name, column)

    @typecheck_method(item=oneof(strlike, Expression, slice, tupleof(Expression)))
    def __getitem__(self, item):
        if isinstance(item, str) or isinstance(item, unicode):
            return self._get_field(item)
        elif isinstance(item, slice):
            s = item
            if not (s.start is None and s.stop is None and s.step is None):
                raise ExpressionException(
                    "Expect unbounded slice syntax ':' to indicate global table join, found unexpected attributes {}".format(
                        ', '.join(x for x in ['start' if s.start is not None else None,
                                              'stop' if s.stop is not None else None,
                                              'step' if s.step is not None else None] if x is not None)
                    )
                )

            return self.index_globals()
        else:
            exprs = item if isinstance(item, tuple) else (item,)
            return self.index_rows(*exprs)

    @property
    @handle_py4j
    def schema(self):
        if self._schema is None:
            self._schema = Type._from_java(self._jkt.signature())
            assert (isinstance(self._schema, TStruct))
        return self._schema

    @property
    @handle_py4j
    def columns(self):
        if self._column_names is None:
            self._column_names = list(self._jkt.columns())
        return self._column_names

    @property
    @handle_py4j
    def num_columns(self):
        if self._num_columns is None:
            self._num_columns = self._jkt.nColumns()
        return self._num_columns

    @handle_py4j
    def count(self):
        return self._jkt.count()

    @classmethod
    @handle_py4j
    @record_classmethod
    @typecheck_method(rows=oneof(listof(Struct), listof(dictof(strlike, anytype))),
                      schema=TStruct,
                      key=oneof(strlike, listof(strlike)),
                      num_partitions=nullable(integral))
    def parallelize(cls, rows, schema, key=[], num_partitions=None):
        return Table(
            Env.hc(),
            Env.hail().table.Table.parallelize(
                Env.hc()._jhc, [schema._convert_to_j(r) for r in rows],
                schema._jtype, wrap_to_list(key), joption(num_partitions)))

    @handle_py4j
    @typecheck_method(keys=tupleof(strlike))
    def key_by(self, *keys):
        """Change which columns are keys.

        **Examples**

        Assume ``kt`` is a :py:class:`.KeyTable` with three columns: c1, c2 and
        c3 and key c1.

        Change key columns:

        >>> table_result = table1.key_by('C2', 'C3')

        >>> table_result = table1.key_by('C2')

        Set to no keys:

        >>> table_result = table1.key_by()

        :param key: List of columns to be used as keys.
        :type key: str or list of str

        :return: Key table whose key columns are given by ``key``.
        :rtype: :class:`.KeyTable`
        """

        return Table(self._hc, self._jkt.keyBy(list(keys)))

    @handle_py4j
    def annotate_globals(self, **named_exprs):
        """Add new global fields.

        Examples
        --------

        Add a new global field:

        >>> table_result = table1.annotate(pops = ['EUR', 'AFR', 'EAS', 'SAS'])

        Parameters
        ----------
        named_exprs : varargs of :class:`hail.expr.expression.Expression`
            Annotation expressions.

        Returns
        -------
        :class:`Table`
            Table with new global field(s).
        """

        exprs = []
        named_exprs = {k: to_expr(v) for k, v in named_exprs.items()}
        base, cleanup = self._process_joins(*named_exprs.values())
        for k, v in named_exprs.items():
            analyze(v, self._global_indices, set(), {f.name for f in self.global_schema.fields})
            exprs.append('`{k}` = {v}'.format(k=k, v=v._ast.to_hql()))

        m = Table(self._hc, base._jkt.annotateGlobalExpr(",\n".join(exprs)))
        return cleanup(m)

    @handle_py4j
    def select_globals(self, *exprs, **named_exprs):
        """Select existing global fields or create new fields by name, dropping the rest.

        Examples
        --------
        Select one existing field and compute a new one:

        >>> table_result = table1.select_globals(table1.global_field_1,
        ...                                      another_global=['AFR', 'EUR', 'EAS', 'AMR', 'SAS'])

        Notes
        -----
        This method creates new global fields. If a created field shares its name
        with a row-indexed field of the table, the method will fail.

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
        :class:`Table`
            Table with specified global fields.
        """

        exprs = tuple(self[e] if not isinstance(e, Expression) else e for e in exprs)
        named_exprs = {k: to_expr(v) for k, v in named_exprs.items()}
        strs = []
        all_exprs = []
        base, cleanup = self._process_joins(*(exprs + tuple(named_exprs.values())))

        for e in exprs:
            all_exprs.append(e)
            analyze(e, self._global_indices, set(), set(f.name for f in self.global_schema.fields))
            if e._ast.search(lambda ast: not isinstance(ast, Reference) and not isinstance(ast, Select)):
                raise ExpressionException("method 'select_globals' expects keyword arguments for complex expressions")
            strs.append(e._ast.to_hql())
        for k, e in named_exprs.items():
            all_exprs.append(e)
            analyze(e, self._global_indices, set(), set(f.name for f in self.global_schema.fields))
            strs.append('`{}` = {}'.format(k, to_expr(e)._ast.to_hql()))

        return cleanup(Table(self._hc, base._jkt.selectGlobal(strs)))

    @handle_py4j
    @typecheck_method(named_exprs=dictof(strlike, anytype))
    def annotate(self, **named_exprs):
        """Add new columns.

        **Examples**

        Add new column ``Y`` which is equal to 5 times ``X``:

        >>> table_result = table1.annotate(Y = 5 * table1.X)

        Add multiple columns simultaneously:

        >>> table_result = table1.annotate(A = table1.X / 2,
        ...                          B = table1.X + 21)

        :param kwargs: Annotation expression with the left hand side equal to the new column name and the right hand side is any type.
        :type kwargs: dict of str to anytype

        :return: Key table with new columns specified by ``named_exprs``.
        :rtype: :class:`.KeyTable`
        """
        named_exprs = {k: to_expr(v) for k, v in named_exprs.items()}
        exprs = []
        base, cleanup = self._process_joins(*named_exprs.values())
        for k, v in named_exprs.items():
            analyze(v, self._row_indices, set(), set(self.columns))
            exprs.append('{k} = {v}'.format(k=k, v=v._ast.to_hql()))

        return cleanup(Table(self._hc, base._jkt.annotate(",\n".join(exprs))))

    @handle_py4j
    @typecheck_method(expr=anytype,
                      keep=bool)
    def filter(self, expr, keep=True):
        """Filter rows.

        Examples
        --------

        Keep rows where ``C1`` equals 5:

        >>> table_result = table1.filter(table1.C1 == 5)

        Remove rows where ``C1`` equals 10:

        >>> table_result = table1.filter(table1.C1 == 10, keep=False)

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
        This method does not support aggregation.

        Parameters
        ----------
        expr : bool or :class:`hail.expr.expression.BooleanExpression`
            Filter expression.
        keep : bool
            Keep rows where `expr` is true.

        Returns
        -------
        :class:`Table`
            Filtered table.
        """
        expr = to_expr(expr)
        analyze(expr, self._row_indices, set(), set(self.columns))
        base, cleanup = self._process_joins(expr)
        if not isinstance(expr._type, TBoolean):
            raise TypeError("method 'filter' expects an expression of type 'TBoolean', found {}"
                            .format(expr._type.__class__))

        return cleanup(Table(self._hc, base._jkt.filter(expr._ast.to_hql(), keep)))

    @handle_py4j
    @typecheck_method(exprs=tupleof(oneof(Expression, strlike)),
                      named_exprs=dictof(strlike, anytype))
    def select(self, *exprs, **named_exprs):
        """Select existing fields or create new fields by name, dropping the rest.

        Examples
        --------
        Select a few old columns and compute a new one:

        >>> table_result = table1.select(table1.ID, table1.C1, Y=table1.Z - table1.X)

        Notes
        -----
        This method creates new row-indexed fields. If a created field shares its name
        with a global field of the table, the method will fail.

        Note
        ----

        **Using select**

        Select and its sibling methods (:meth:`Table.select_globals`,
        :meth:`MatrixTable.select_globals`, :meth:`MatrixTable.select_rows`,
        :meth:`MatrixTable.select_cols`, and :meth:`MatrixTable.select_entries`) accept
        both variable-length (``f(x, y, z)``) and keyword (``f(a=x, b=y, c=z)``)
        arguments.

        Variable-length arguments can be either strings or expressions that reference a
        (possibly nested) field of the table. Keyword arguments can be arbitrary
        expressions.

        **The following three usages are all equivalent**, producing a new table with
        columns `C1` and `C2` of `table1`.

        First, variable-length string arguments:

        >>> table_result = table1.select('C1', 'C2')

        Second, field reference variable-length arguments:

        >>> table_result = table1.select(table1.C1, table1.C2)

        Last, expression keyword arguments:

        >>> table_result = table1.select(C1 = table1.C1, C2 = table1.C2)

        Additionally, the variable-length argument syntax also permits nested field
        references. Given the following struct field `s`:

        >>> table3 = table1.annotate(s = Struct(x=table1.X, z=table1.Z))

        The following two usages are equivalent, producing a table with one field, `x`.:

        >>> table3_result = table3.select(table3.s.x)

        >>> table3_result = table3.select(x = table3.s.x)

        The keyword argument syntax permits arbitrary expressions:

        >>> table_result = table1.select(foo=table1.X ** 2 + 1)

        These syntaxes can be mixed together, with the stipulation that all keyword arguments
        must come at the end due to Python language restrictions.

        >>> table_result = table1.select(table1.X, 'Z', bar = [table1.C1, table1.C2])

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
        :class:`Table`
            Table with specified fields.
        """
        exprs = tuple(self[e] if not isinstance(e, Expression) else e for e in exprs)
        named_exprs = {k: to_expr(v) for k, v in named_exprs.items()}
        strs = []
        all_exprs = []
        base, cleanup = self._process_joins(*(exprs + tuple(named_exprs.values())))

        for e in exprs:
            all_exprs.append(e)
            analyze(e, self._row_indices, set(), set(self.columns))
            if e._ast.search(lambda ast: not isinstance(ast, Reference) and not isinstance(ast, Select)):
                raise ExpressionException("method 'select' expects keyword arguments for complex expressions")
            strs.append(e._ast.to_hql())
        for k, e in named_exprs.items():
            all_exprs.append(e)
            analyze(e, self._row_indices, set(), set(self.columns))
            strs.append('`{}` = {}'.format(k, to_expr(e)._ast.to_hql()))

        return cleanup(Table(self._hc, base._jkt.select(strs, False)))

    @handle_py4j
    @typecheck_method(exprs=tupleof(oneof(strlike, Expression)))
    def drop(self, *exprs):
        """Drop fields from the table.

        Examples
        --------

        Drop fields `C1` and `C2` using strings:

        >>> table_result = table1.drop('C1', 'C2')

        Drop fields `C1` and `C2` using field references:

        >>> table_result = table1.drop(table1.C1, table1.C2)

        Drop a list of fields:

        >>> fields_to_drop = ['C1', 'C2']
        >>> table_result = table1.drop(*fields_to_drop)

        Notes
        -----

        This method can be used to drop global or row-indexed fields. The arguments
        can be either strings (``'field'``), or top-level field references
        (``table.field`` or ``table['field']``).

        Parameters
        ----------
        exprs : varargs of :obj:`str` or :class:`hail.expr.expression.Expression`
            Names of fields to drop or field reference expressions.

        Returns
        -------
        :class:`Table`
            Table without specified fields.
        """
        all_field_exprs = {e: k for k, e in self._fields.items()}
        fields_to_drop = set()
        for e in exprs:
            if isinstance(e, Expression):
                if e in all_field_exprs:
                    fields_to_drop.add(all_field_exprs[e])
                else:
                    raise ExpressionException("method 'drop' expects string field names or top-level field expressions"
                                              " (e.g. table['foo'])")
            else:
                assert isinstance(e, str) or isinstance(str, unicode)
                if e not in self._fields:
                    raise IndexError("table has no field '{}'".format(e))
                fields_to_drop.add(e)

        table = self
        if any(self._fields[field]._indices == self._global_indices for field in fields_to_drop):
            # need to drop globals
            new_global_fields = {k.name: table._fields[k.name] for k in table.global_schema.fields if
                                 k.name not in fields_to_drop}
            table = table.select_globals(**new_global_fields)

        if any(self._fields[field]._indices == self._row_indices for field in fields_to_drop):
            # need to drop row fields
            new_row_fields = {k.name: table._fields[k.name] for k in table.schema.fields if
                              k.name not in fields_to_drop}
            table = table.select(**new_row_fields)

        return table

    @handle_py4j
    def export(self, output, types_file=None, header=True, parallel=None):
        """Export to a TSV file.

        Examples
        --------
        Export to a tab-separated file:

        >>> table1.export('output/table1.tsv.bgz')

        Note
        ----
        It is highly recommended to export large files with a ``.bgz`` extension,
        which will use a block gzipped compression codec. These files can be
        read natively with any Hail method, as well as with Python's ``gzip.open``
        and R's ``read.table``.

        Parameters
        ----------
        output : str
            URI at which to write exported file.
        types_file : str or None
            URI at which to write file containing column type information.
        header : bool
            Include a header in the file.
        parallel : bool
            Skip the merge step after the parallel write, producing a directory instead
            of a single file.
        """
        self._jkt.export(output, types_file, header, parallel)

    @typecheck_method(exprs=tupleof(anytype),
                      named_exprs=dictof(strlike, anytype))
    def group_by(self, *exprs, **named_exprs):
        """Group by a new set of keys for use with :meth:`GroupedTable.aggregate`.

        Examples
        --------
        Compute the mean value of `X` and the sum of `Z` per unique `ID`:

        >>> table_result = (table1.group_by(table1.ID)
        ...                       .aggregate(meanX = f.mean(table1.X), sumZ = f.sum(table1.Z)))

        Group by a height bin and compute sex ratio per bin:

        >>> table_result = (table1.group_by(height_bin = (table1.HT / 20).to_int32())
        ...                       .aggregate(fraction_female = f.fraction(table1.SEX == 'F')))

        Notes
        -----
        This function is always followed by :meth:`GroupedTable.aggregate`. Follow the
        link for documentation on the aggregation step.

        Note
        ----
        **Using group_by**

        **group_by** and its sibling methods (:meth:`MatrixTable.group_rows_by` and
        :meth:`MatrixTable.group_cols_by` accept both variable-length (``f(x, y, z)``)
        and keyword (``f(a=x, b=y, c=z)``) arguments.

        Variable-length arguments can be either strings or expressions that reference a
        (possibly nested) field of the table. Keyword arguments can be arbitrary
        expressions.

        **The following three usages are all equivalent**, producing a
        :class:`GroupedTable` grouped by columns `C1` and `C2` of `table1`.

        First, variable-length string arguments:

        >>> table_result = (table1.group_by('C1', 'C2')
        ...                       .aggregate(meanX = f.mean(table1.X)))

        Second, field reference variable-length arguments:

        >>> table_result = (table1.group_by(table1.C1, table1.C2)
        ...                       .aggregate(meanX = f.mean(table1.X)))

        Last, expression keyword arguments:

        >>> table_result = (table1.group_by(C1 = table1.C1, C2 = table1.C2)
        ...                       .aggregate(meanX = f.mean(table1.X)))

        Additionally, the variable-length argument syntax also permits nested field
        references. Given the following struct field `s`:

        >>> table3 = table1.annotate(s = Struct(x=table1.X, z=table1.Z))

        The following two usages are equivalent, grouping by one field, `x`:

        >>> table_result = (table3.group_by(table3.s.x)
        ...                       .aggregate(meanX = f.mean(table3.X)))

        >>> table_result = (table3.group_by(x = table3.s.x)
        ...                       .aggregate(meanX = f.mean(table3.X)))

        The keyword argument syntax permits arbitrary expressions:

        >>> table_result = (table1.group_by(foo=table1.X ** 2 + 1)
        ...                       .aggregate(meanZ = f.mean(table1.Z)))

        These syntaxes can be mixed together, with the stipulation that all keyword arguments
        must come at the end due to Python language restrictions.

        >>> table_result = (table1.group_by(table1.C1, 'C2', height_bin = (table1.HT / 20).to_int32())
        ...                       .aggregate(meanX = f.mean(table1.X)))

        Note
        ----
        This method does not support aggregation in key expressions.

        Arguments
        ---------
        exprs : varargs of type str or :class:`hail.expr.expression.Expression`
            Field names or field reference expressions.
        named_exprs : keyword args of type :class:`hail.expr.expression.Expression`
            Field names and expressions to compute them.

        Returns
        -------
        :class:`GroupedTable`
            Grouped table; use :meth:`GroupedTable.aggregate` to complete the aggregation.
        """
        groups = []
        for e in exprs:
            if isinstance(e, str) or isinstance(e, unicode):
                e = self[e]
            else:
                e = to_expr(e)
            analyze(e, self._row_indices, set(), set(self.columns))
            ast = e._ast.expand()
            if any(not isinstance(a, Reference) and not isinstance(a, Select) for a in ast):
                raise ExpressionException("method 'group_by' expects keyword arguments for complex expressions")
            key = ast[0].name if isinstance(ast[0], Reference) else ast[0].selection
            groups.append((key, e))
        for k, e in named_exprs.items():
            e = to_expr(e)
            analyze(e, self._row_indices, set(), set(self.columns))
            groups.append((k, e))

        return GroupedTable(self, groups)

    @handle_py4j
    def aggregate(self, **named_exprs):
        """Aggregate over rows into a local struct.

        Examples
        --------
        Aggregate over rows:

        .. doctest::

            >>> table1.aggregate(fraction_male = f.fraction(table1.SEX == 'M'),
            ...                  mean_x = f.mean(table1.X))
            Struct(fraction_male=0.5, mean_x=6.5)

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
        agg_base = self.columns[0]  # FIXME hack

        named_exprs = {k: to_expr(v) for k, v in named_exprs.items()}
        strs = []
        base, _ = self._process_joins(*named_exprs.values())
        for k, v in named_exprs.items():
            analyze(v, self._global_indices, {self._row_axis}, set(self.columns))
            replace_aggregables(v._ast, agg_base)
            strs.append(v._ast.to_hql())

        result_list = base._jkt.query(jarray(Env.jvm().java.lang.String, strs))
        ptypes = [Type._from_java(x._2()) for x in result_list]

        annotations = [ptypes[i]._convert_to_py(result_list[i]._1()) for i in range(len(ptypes))]
        d = {k: v for k, v in zip(named_exprs.keys(), annotations)}
        return Struct(**d)

    @handle_py4j
    @typecheck_method(output=strlike,
                      overwrite=bool)
    def write(self, output, overwrite=False):
        """Write to disk.

        Examples
        --------

        >>> table1.write('output/table1.kt')

        Note
        ----
        The write path must end in ".kt".

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

        self._jkt.write(output, overwrite)

    @handle_py4j
    @typecheck_method(n=integral, truncate_to=nullable(integral), print_types=bool)
    def show(self, n=10, truncate_to=None, print_types=True):
        """Print the first few rows of the table to the console.

        Examples
        --------
        Show the first 20 lines:

        >>> table1.show(20)

        Parameters
        ----------
        n : int
            Maximum number of rows to show.
        truncate_to : bool
            Truncate each column to the given number of characters
        print_types : bool
            Print an extra header line with the type of each field.
        """
        return self.to_hail1().show(n, truncate_to, print_types)

    def to_hail1(self):
        """Convert table to :class:`hail.api1.KeyTable`.

        Returns
        -------
        :class:`hail.api1.KeyTable`
        """
        import hail
        kt = hail.KeyTable(self._hc, self._jkt)
        kt._set_history(History('is a mystery'))
        return kt

    @handle_py4j
    def index_rows(self, *exprs):
        if not len(exprs) > 0:
            raise ValueError('Require at least one expression to index a table')

        exprs = [to_expr(e) for e in exprs]
        if not len(exprs) == len(self.key):
            raise ExpressionException('Key mismatch: table has {} keys, found {} index expressions'.format(
                len(self.key), len(exprs)))

        i = 0
        for k, e in zip(self.key, exprs):
            if not self[k]._type == e._type:
                raise ExpressionException(
                    'Type mismatch at index {} of Table index: expected key type {}, found {}'.format(
                        i, str(self[k]._type), str(e._type)))
            i += 1

        indices, aggregations, joins = unify_all(*exprs)

        from hail.api2.matrixtable import MatrixTable
        uid = Env._get_uid()

        src = indices.source
        if src is None or len(indices.axes) == 0:
            # FIXME: this should be OK: table[m.global_index_into_table]
            raise ExpressionException('found explicit join indexed by a scalar expression')
        elif isinstance(src, Table):
            for e in exprs:
                analyze(e, src._row_indices, set(), set(src.columns))

            right = self
            right_keys = [right[k] for k in right.key]
            select_struct = Struct(**{k: right[k] for k in right.columns})
            right = right.select(*right_keys, **{uid: select_struct})
            uids = [Env._get_uid() for i in range(len(exprs))]
            full_key_strs = ',\n'.join('{}={}'.format(uids[i], exprs[i]._ast.to_hql()) for i in range(len(exprs)))

            def joiner(left):
                left = Table(self._hc, left._jkt.annotate(full_key_strs)).key_by(*uids)
                left = left.to_hail1().join(right.to_hail1(), 'left').to_hail2()
                return left

            all_uids = uids[:]
            all_uids.append(uid)
            return construct_expr(Reference(uid), self.schema, indices, aggregations,
                                           joins + (Join(joiner, all_uids),))
        elif isinstance(src, MatrixTable):
            for e in exprs:
                analyze(e, src._entry_indices, set(), set(src._fields.keys()))

            right = self
            # match on indices to determine join type
            if indices == src._entry_indices:
                raise NotImplementedError('entry-based matrix joins')
            elif indices == src._row_indices:
                if len(exprs) == 1 and exprs[0] is src['v']:
                    # no vds_key (way faster)
                    joiner = lambda left: MatrixTable(self._hc, left._jvds.annotateVariantsTable(
                        right._jkt, None, 'va.{}'.format(uid), None, False))
                else:
                    # use vds_key
                    joiner = lambda left: MatrixTable(self._hc, left._jvds.annotateVariantsTable(
                        right._jkt, [e._ast.to_hql() for e in exprs], 'va.{}'.format(uid), None, False))

                return construct_expr(Select(Reference('va'), uid), self.schema,
                               indices, aggregations, joins + (Join(joiner, [uid]),))
            elif indices == src._col_indices:
                if len(exprs) == 1 and exprs[0] is src['s']:
                    # no vds_key (faster)
                    joiner = lambda left: MatrixTable(self._hc, left._jvds.annotateSamplesTable(
                        right._jkt, None, 'sa.{}'.format(uid), None, False))
                else:
                    # use vds_key
                    joiner = lambda left: MatrixTable(self._hc, left._jvds.annotateSamplesTable(
                        right._jkt, [e._ast.to_hql() for e in exprs], 'sa.{}'.format(uid), None, False))
                return construct_expr(Select(Reference('sa'), uid), self.schema,
                               indices, aggregations, joins + (Join(joiner, [uid]),))
            else:
                raise NotImplementedError()
        else:
            raise TypeError("Cannot join with expressions derived from '{}'".format(src.__class__))

    @handle_py4j
    def index_globals(self):
        uid = Env._get_uid()

        def joiner(obj):
            from hail.api2.matrixtable import MatrixTable
            if isinstance(obj, MatrixTable):
                return MatrixTable(obj._hc, Env.jutils().joinGlobals(obj._jvds, self._jkt, uid))
            else:
                assert isinstance(obj, Table)
                return Table(obj._hc, Env.jutils().joinGlobals(obj._jkt, self._jkt, uid))

        return construct_expr(GlobalJoinReference(uid), self.global_schema, joins=(Join(joiner, [uid]),))

    @typecheck_method(exprs=tupleof(Expression))
    def _process_joins(self, *exprs):
        # ordered to support nested joins
        original_key = self.key

        all_uids = []
        left = self

        for e in exprs:
            rewrite_global_refs(e._ast, self)
            for j in e._joins:
                left = j.join_function(left)
                all_uids.extend(j.temp_vars)

        if left is not self:
            left = left.key_by(*original_key)

        def cleanup(table):
            return table.drop(*all_uids)

        return left, cleanup

    @classmethod
    @handle_py4j
    @typecheck_method(n=integral,
                      num_partitions=nullable(integral))
    def range(cls, n, num_partitions=None):
        """Construct a table with `n` rows with field `index` that ranges from 0 to ``n - 1``.

        Examples
        --------
        Construct a table with 100 rows:

        >>> range_table = Table.range(100)

        Construct a table with one million rows and twenty partitions:

        >>> range_table = Table.range(1000000, 20)

        Notes
        -----
        The resulting table has one column:

         - **index** (`Int`) - Unique row index from 0 to ``n - 1``

        Parameters
        ----------
        n : int
            Number of rows.
        num_partitions : int
            Number of partitions

        Returns
        -------
        :class:`Table`
            Table with one field, `index`.
        """
        return Table(Env.hc(), Env.hail().table.Table.range(Env.hc()._jhc, n, joption(num_partitions)))

    @handle_py4j
    def cache(self):
        """Persist this table in memory.

        Examples
        --------
        Persist the table in memory:

        >>> table = table.cache() # doctest: +SKIP

        Notes
        -----

        This method is an alias for :func:`persist("MEMORY_ONLY") <hail.api2.Table.persist>`.

        Returns
        -------
        :class:`Table`
            Cached table.
        """
        return self.persist('MEMORY_ONLY')


    def persist(self, storage_level='MEMORY_AND_DISK'):
        """Persist this table in memory or on disk.

        Examples
        --------
        Persist the key table to both memory and disk:

        >>> table = table.persist() # doctest: +SKIP

        Notes
        -----

        The :py:meth:`Table.persist` and :py:meth:`Table.cache` methods store the
        current table on disk or in memory temporarily to avoid redundant computation
        and improve the performance of Hail pipelines. This method is not a substitution
        for :py:meth:`Table.write`, which stores a permanent file.

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
        :class:`Table`
            Persisted table.
        """
        return Table(self._hc, self._jkt.persist(storage_level))

    @handle_py4j
    def collect(self):
        """Collect the rows of the table into a local list.

        Examples
        --------
        Collect a list of all `X` records:

        >>> all_xs = [row['X'] for row in table1.select(table1.X).collect()]

        Notes
        -----
        This method returns a list whose elements are of type :class:`Struct`. Fields
        of these structs can be accessed similarly to fields on a table, using dot
        methods (``struct.foo``) or string indexing (``struct['foo']``).

        Warning
        -------
        Using this method can cause out of memory errors. Only collect small tables.

        Returns
        -------
        :obj:`list` of :class:`Struct`
            List of rows.
        """
        return TArray(self.schema)._convert_to_py(self._jkt.collect())

    @typecheck_method(truncate_at=integral)
    def describe(self, truncate_at=60):
        """Print information about the fields in the table."""
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

        key_set = set(self.key)
        if len(self.schema.fields) == 0:
            row_fields = '\n    None'
        else:
            row_fields = ''.join("\n    '{name}'{is_key}: {type} ".format(
                name=fd.name, is_key='' if fd.name not in key_set else ' [key field]',
                type=format_type(fd.typ)) for fd in self.schema.fields)

        s = 'Global fields:{}\n\nRow fields:{}'.format(global_fields, row_fields)
        print(s)
