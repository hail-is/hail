from __future__ import print_function  # Python 2 and 3 print compatibility

from hail2.expr.expression import *
import hail2.expr.functions as f
from hail.java import *
from hail.typ import Type, TArray, TStruct
from hail.representation import Struct
from hail.typecheck import *
from hail.utils import wrap_to_list
from pyspark.sql import DataFrame
from hail.history import *
from hail.typecheck import *


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
            raise KeyError("No field '{name} found. "
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
    """KeyTable that has been grouped.
    """

    def __init__(self, parent, groups):
        super(GroupedTable, self).__init__(parent._hc, parent._jkt)
        self._groups = groups
        self._parent = parent
        self._npartitions = None

        for fd in parent._fields:
            self._set_field(fd, parent._fields[fd])

    @property
    def groups(self):
        return self._groups

    @handle_py4j
    @typecheck_method(n=integral)
    def set_partitions(self, n):
        self._npartitions = n
        return self

    @handle_py4j
    @typecheck_method(named_exprs=dictof(strlike, anytype))
    def aggregate(self, **named_exprs):
        """Aggregate columns programmatically by key.

        :param named_exprs: Annotation expression with the left hand side equal to the new column name and the right hand side is any type.
        :type named_exprs: dict of str to anytype

        :return: Key table with new columns specified by ``kwargs`` that have been aggregated by the key specified by groups.
        :rtype: :class:`.KeyTable`
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
    """Hail's version of a SQL table where columns can be designated as keys.

    Key tables may be imported from a text file or Spark DataFrame with :py:meth:`~hail2.HailContext.import_table`
    or :py:meth:`~hail2.KeyTable.from_dataframe`, generated from a variant dataset
    with :py:meth:`~hail2.VariantDataset.make_table`, :py:meth:`~hail2.VariantDataset.genotypes_table`,
    :py:meth:`~hail2.VariantDataset.samples_table`, or :py:meth:`~hail2.VariantDataset.variants_table`.

    In the examples below, we have imported two key tables from text files (``kt1`` and ``kt2``).

    .. testsetup::

        hc.stop()
        import hail2 as h2
        from hail2 import *
        hc = h2.HailContext()

    >>> table1 = hc.import_table('data/kt_example1.tsv', impute=True)
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

    >>> table2 = hc.import_table('data/kt_example2.tsv', impute=True)
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

    Filter rows of the table:

    >>> table2 = table2.filter(table2.B != 'rabbit')

    Compute global aggregation statistics:

    >>> t1_stats = table1.aggregate(mean_c1 = f.mean(table1.C1),
    ...                             mean_c2 = f.mean(table1.C2),
    ...                             stats_c3 = f.stats(table1.C3))
    >>> print(t1_stats)

    Group columns and aggregate to produce a new table:

    >>> table3 = table1.group_by(table1.SEX)\
    ...                .aggregate(mean_height_data = f.mean(table1.HT))
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
            column = convert_expr(
                Expression(Reference(fd.name), fd.typ, indices=self._global_indices, aggregations=(), joins=()))
            self._set_field(fd.name, column)

        for fd in self.schema.fields:
            column = convert_expr(
                Expression(Reference(fd.name), fd.typ, indices=self._row_indices, aggregations=(), joins=()))
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
            Env.hail().keytable.KeyTable.parallelize(
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

        >>> kt_result = kt1.key_by('C2', 'C3')

        >>> kt_result = kt1.key_by('C2')

        Set to no keys:

        >>> kt_result = kt1.key_by()

        :param key: List of columns to be used as keys.
        :type key: str or list of str

        :return: Key table whose key columns are given by ``key``.
        :rtype: :class:`.KeyTable`
        """

        return Table(self._hc, self._jkt.keyBy(list(keys)))

    @handle_py4j
    def annotate_globals(self, **named_exprs):
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

        >>> kt_result = kt1.annotate(Y = 5 * kt1.X)

        Add multiple columns simultaneously:

        >>> kt_result = kt1.annotate(A = kt1.X / 2,
        ...                          B = kt1.X + 21)

        :param kwargs: Annotation expression with the left hand side equal to the new column name and the right hand side is any type.
        :type kwargs: dict of str to anytype

        :return: Key table with new columns specified by ``named_exprs``.
        :rtype: :class:`.KeyTable`
        """

        # ordered to support nested joins
        unique_join_ids = OrderedDict()

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

        **Examples**

        Keep rows where ``C1`` equals 5:

        >>> kt_result = kt1.filter(kt1.C1 == 5)

        Remove rows where ``C1`` equals 10:

        >>> kt_result = kt1.filter(kt1.C1 == 10, keep=False)

        **Notes**

        The scope for ``expr`` is all column names in the input :class:`KeyTable`.

        .. caution::
           When ``expr`` evaluates to missing, the row will be removed regardless of whether ``keep=True`` or ``keep=False``.

        :param expr: Boolean filter expression.
        :type expr: :class:`~hail2.expr.column.BooleanColumn` or bool

        :param bool keep: Keep rows where ``expr`` is true.

        :return: Filtered key table.
        :rtype: :class:`.KeyTable`
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
        """Select a subset of columns.

        **Examples**

        Assume ``kt1`` is a :py:class:`.KeyTable` with three columns: C1, C2 and
        C3.

        Select/drop columns:

        >>> kt_result = kt1.select(kt1.C1)

        Reorder the columns:

        >>> kt_result = kt1.select(kt1.C3, kt1.C1, kt1.C2)

        Drop all columns:

        >>> kt_result = kt1.select()

        Create a new column computed from existing columns:

        >>> kt_result = kt1.select(C_NEW = kt1.C1 + kt1.C2 + kt1.C3)

        :return: Key table with selected columns.
        :rtype: :class:`.KeyTable`
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
        """Drop fields from the table."""

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
    def export(self, output, types_file=None, header=True, parallel=False):
        """Export to a TSV file.

        :param output:
        :param types_file:
        :param header:
        """
        self._jkt.export(output, types_file, header, parallel)

    @typecheck_method(exprs=tupleof(anytype),
                      named_exprs=dictof(strlike, anytype))
    def group_by(self, *exprs, **named_exprs):
        """Group by key.

        :return: Key table where groupings are computed from expressions given by `kwargs`.
        :rtype: :class:`.GroupedKeyTable`
        """
        groups = []
        for e in exprs:
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
        """Hail2's version of the old 'query' """
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
        """Write as KT file.

        ***Examples***

        >>> kt1.write('output/kt1.kt')

        .. note:: The write path must end in ".kt".

        **Notes**

        A text file containing the python code to generate this output file is available at ``<output>/history.txt``.

        :param str output: Path of KT file to write.

        :param bool overwrite: If True, overwrite any existing KT file. Cannot be used
               to read from and write to the same path.
        """

        self._jkt.write(output, overwrite)

    @handle_py4j
    def show(self, n=10, truncate_to=None, print_types=True):
        return self.to_hail1().show(n, truncate_to, print_types)

    def to_hail1(self):
        """Convert table to :class:`hail.KeyTable`.

        :rtype: :class:`hail.KeyTable`
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

        from hail2.matrix import Matrix
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
            return convert_expr(Expression(Reference(uid), self.schema, indices, aggregations,
                                           joins + (Join(joiner, all_uids),)))
        elif isinstance(src, Matrix):
            for e in exprs:
                analyze(e, src._entry_indices, set(), set(src._fields.keys()))

            right = self
            # match on indices to determine join type
            if indices == src._entry_indices:
                raise NotImplementedError('entry-based matrix joins')
            elif indices == src._row_indices:
                if len(exprs) == 1 and exprs[0] is src['v']:
                    # no vds_key (way faster)
                    joiner = lambda left: Matrix(self._hc, left._jvds.annotateVariantsTable(
                        right._jkt, None, 'va.{}'.format(uid), None, False))
                else:
                    # use vds_key
                    joiner = lambda left: Matrix(self._hc, left._jvds.annotateVariantsTable(
                        right._jkt, [e._ast.to_hql() for e in exprs], 'va.{}'.format(uid), None, False))

                return convert_expr(
                    Expression(Select(Reference('va'), uid), self.schema,
                               indices, aggregations, joins + (Join(joiner, [uid]),)))
            elif indices == src._col_indices:
                if len(exprs) == 1 and exprs[0] is src['s']:
                    # no vds_key (faster)
                    joiner = lambda left: Matrix(self._hc, left._jvds.annotateSamplesTable(
                        right._jkt, None, 'sa.{}'.format(uid), None, False))
                else:
                    # use vds_key
                    joiner = lambda left: Matrix(self._hc, left._jvds.annotateSamplesTable(
                        right._jkt, [e._ast.to_hql() for e in exprs], 'sa.{}'.format(uid), None, False))
                return convert_expr(
                    Expression(Select(Reference('sa'), uid), self.schema,
                               indices, aggregations, joins + (Join(joiner, [uid]),)))
            else:
                raise NotImplementedError()
        else:
            raise TypeError("Cannot join with expressions derived from '{}'".format(src.__class__))

    @handle_py4j
    def index_globals(self):
        uid = Env._get_uid()

        def joiner(obj):
            from hail2.matrix import Matrix
            if isinstance(obj, Matrix):
                return Matrix(obj._hc, Env.jutils().joinGlobals(obj._jvds, self._jkt, uid))
            else:
                assert isinstance(obj, Table)
                return Table(obj._hc, Env.jutils().joinGlobals(obj._jkt, self._jkt, uid))

        return convert_expr(
            Expression(GlobalJoinReference(uid), self.global_schema, Indices(source=self), (), (Join(joiner, [uid]),)))

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
        """Construct a table of ``n`` rows with values 0 to ``n - 1``.

        **Examples**

        Construct a table with 100 rows:

        >>> range_kt = Table.range(100)

        Construct a table with one million rows and twenty partitions:

        >>> range_kt = Table.range(1000000, num_partitions=20)

        **Notes**

        The resulting table has one column:

         - **index** (*Int*) -- Unique row index from 0 to ``n - 1``

        :param int n: Number of rows.

        :param num_partitions: Number of partitions.
        :type num_partitions: int or None

        :rtype: :class:`.KeyTable`
        """

        return Table(Env.hc(), Env.hail().keytable.KeyTable.range(Env.hc()._jhc, n, joption(num_partitions)))

    def persist(self, storage_level='MEMORY_AND_DISK'):
        return Table(self._hc, self._jkt.persist(storage_level))

    @handle_py4j
    def collect(self):
        return TArray(self.schema)._convert_to_py(self._jkt.collect())
