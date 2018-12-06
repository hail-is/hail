import pandas
import pyspark
import warnings

from typing import *

import hail as hl
from hail.expr.expressions import *
from hail.expr.types import *
from hail.ir import *
from hail.typecheck import *
from hail.utils import wrap_to_list, storage_level, LinkedList, Struct
from hail.utils.java import *
from hail.utils.misc import *

from collections import OrderedDict, Counter
import itertools

table_type = lazy()


class Ascending(object):
    def __init__(self, col):
        self.col = col

    def _j_obj(self):
        return scala_package_object(Env.hail().table).asc(self.col)


class Descending(object):
    def __init__(self, col):
        self.col = col

    def _j_obj(self):
        return scala_package_object(Env.hail().table).desc(self.col)


@typecheck(col=oneof(Expression, str))
def asc(col):
    """Sort by `col` ascending."""

    return Ascending(col)


@typecheck(col=oneof(Expression, str))
def desc(col):
    """Sort by `col` descending."""

    return Descending(col)

class ExprContainer(object):

    # this can only grow as big as the object dir, so no need to worry about memory leak
    _warned_about = set()

    def __init__(self):
        self._fields: Dict[str, Expression] = {}
        self._fields_inverse: Dict[Expression, str] = {}
        super(ExprContainer, self).__init__()

    def _set_field(self, key, value):
        assert key not in self._fields_inverse, key
        self._fields[key] = value
        self._fields_inverse[value] = key

        # key is in __dir for methods
        # key is in __dict__ for private class fields
        if hasattr(self, key):
            if key not in ExprContainer._warned_about:
                ExprContainer._warned_about.add(key)
                warn(f"Name collision: field {repr(key)} already in object dict. "
                     f"\n  This field must be referenced with __getitem__ syntax: obj[{repr(key)}]")
        else:
            self.__dict__[key] = value

    def _get_field(self, item) -> Expression:
        if item in self._fields:
            return self._fields[item]
        else:
            raise LookupError(get_nice_field_error(self, item))

    def __iter__(self):
        raise TypeError(f"'{self.__class__.__name__}' object is not iterable")

    def __delattr__(self, item):
        if not item[0] == '_':
            raise NotImplementedError(f"'{self.__class__.__name__}' object is not mutable")

    def __setattr__(self, key, value):
        if not key[0] == '_':
            raise NotImplementedError(f"'{self.__class__.__name__}' object is not mutable")
        self.__dict__[key] = value

    def __getattr__(self, item):
        if item in self.__dict__:
            return self.__dict__[item]
        else:
            raise AttributeError(get_nice_attr_error(self, item))

    def _copy_fields_from(self, other: 'ExprContainer'):
        self._fields = other._fields
        self._fields_inverse = other._fields_inverse


class GroupedTable(ExprContainer):
    """Table grouped by row that can be aggregated into a new table.

    There are only two operations on a grouped table, :meth:`.GroupedTable.partition_hint`
    and :meth:`.GroupedTable.aggregate`.
    """

    def __init__(self, parent: 'Table', groups):
        super(GroupedTable, self).__init__()
        self._groups = groups
        self._parent = parent
        self._npartitions = None
        self._buffer_size = 50

        self._copy_fields_from(parent)

    def partition_hint(self, n: int) -> 'GroupedTable':
        """Set the target number of partitions for aggregation.

        Examples
        --------

        Use `partition_hint` in a :meth:`.Table.group_by` / :meth:`.GroupedTable.aggregate`
        pipeline:

        >>> table_result = (table1.group_by(table1.ID)
        ...                       .partition_hint(5)
        ...                       .aggregate(meanX = agg.mean(table1.X), sumZ = agg.sum(table1.Z)))

        Notes
        -----
        Until Hail's query optimizer is intelligent enough to sample records at all
        stages of a pipeline, it can be necessary in some places to provide some
        explicit hints.

        The default number of partitions for :meth:`.GroupedTable.aggregate` is the
        number of partitions in the upstream table. If the aggregation greatly
        reduces the size of the table, providing a hint for the target number of
        partitions can accelerate downstream operations.

        Parameters
        ----------
        n : int
            Number of partitions.

        Returns
        -------
        :class:`.GroupedTable`
            Same grouped table with a partition hint.
        """
        self._npartitions = n
        return self

    def _set_buffer_size(self, n: int) -> 'GroupedTable':
        """Set the map-side combiner buffer size (in rows).

        Parameters
        ----------
        n : int
            Buffer size.

        Returns
        -------
        :class:`.GroupedTable`
            Same grouped table with a buffer size.
        """
        if n <= 0:
            raise ValueError(n)
        self._buffer_size = n
        return self

    @typecheck_method(named_exprs=expr_any)
    def aggregate(self, **named_exprs):
        """Aggregate by group, used after :meth:`.Table.group_by`.

        Examples
        --------
        Compute the mean value of `X` and the sum of `Z` per unique `ID`:

        >>> table_result = (table1.group_by(table1.ID)
        ...                       .aggregate(meanX = agg.mean(table1.X), sumZ = agg.sum(table1.Z)))

        Group by a height bin and compute sex ratio per bin:

        >>> table_result = (table1.group_by(height_bin = table1.HT // 20)
        ...                       .aggregate(fraction_female = agg.fraction(table1.SEX == 'F')))

        Notes
        -----
        The resulting table has a key field for each group and a value field for
        each aggregation. The names of the aggregation expressions must be
        distinct from the names of the groups.

        Parameters
        ----------
        named_exprs : varargs of :class:`.Expression`
            Aggregation expressions.

        Returns
        -------
        :class:`.Table`
            Aggregated table.
        """
        if self._groups is None:
            raise ValueError('GroupedTable cannot be aggregated if no groupings are specified.')

        group_exprs = dict(self._groups)

        for name, expr in named_exprs.items():
            analyze(f'GroupedTable.aggregate: ({repr(name)})', expr, self._parent._global_indices, {self._parent._row_axis})
        if not named_exprs.keys().isdisjoint(group_exprs.keys()):
            intersection = set(named_exprs.keys()) & set(group_exprs.keys())
            raise ValueError(
                f'GroupedTable.aggregate: Group names and aggregration expression names overlap: {intersection}')

        base, _ = self._parent._process_joins(*group_exprs.values(), *named_exprs.values())

        return Table(TableKeyByAndAggregate(base._tir,
                                            hl.struct(**named_exprs)._ir,
                                            hl.struct(**group_exprs)._ir,
                                            self._npartitions,
                                            self._buffer_size))


class Table(ExprContainer):
    """Hail's distributed implementation of a dataframe or SQL table.

    Use :func:`.read_table` to read a table that was written with
    :meth:`.Table.write`. Use :meth:`.to_spark` and :meth:`.Table.from_spark`
    to inter-operate with PySpark's
    `SQL <https://spark.apache.org/docs/latest/sql-programming-guide.html>`__ and
    `machine learning <https://spark.apache.org/docs/latest/ml-guide.html>`__
    functionality.

    Examples
    --------

    The examples below use ``table1`` and ``table2``, which are imported
    from text files using :func:`.import_table`.

    >>> table1 = hl.import_table('data/kt_example1.tsv', impute=True, key='ID')
    >>> table1.show()

    .. code-block:: text

        +-------+-------+-----+-------+-------+-------+-------+-------+
        |    ID |    HT | SEX |     X |     Z |    C1 |    C2 |    C3 |
        +-------+-------+-----+-------+-------+-------+-------+-------+
        | int32 | int32 | str | int32 | int32 | int32 | int32 | int32 |
        +-------+-------+-----+-------+-------+-------+-------+-------+
        |     1 |    65 | M   |     5 |     4 |     2 |    50 |     5 |
        |     2 |    72 | M   |     6 |     3 |     2 |    61 |     1 |
        |     3 |    70 | F   |     7 |     3 |    10 |    81 |    -5 |
        |     4 |    60 | F   |     8 |     2 |    11 |    90 |   -10 |
        +-------+-------+-----+-------+-------+-------+-------+-------+

    >>> table2 = hl.import_table('data/kt_example2.tsv', impute=True, key='ID')
    >>> table2.show()

    .. code-block:: text

        +-------+-------+--------+
        |    ID |     A | B      |
        +-------+-------+--------+
        | int32 | int32 | str    |
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
    ...    return hl.cond(sex == 'M',
    ...                  (height - height_mean_m) / height_sd_m,
    ...                  (height - height_mean_f) / height_sd_f)
    >>>
    >>> table1 = table1.annotate(height_z = get_z(table1.HT, table1.SEX))
    >>> table1 = table1.annotate_globals(global_field_1 = [1, 2, 3])

    Filter rows of the table:

    >>> table2 = table2.filter(table2.B != 'rabbit')

    Compute global aggregation statistics:

    >>> t1_stats = table1.aggregate(hl.struct(mean_c1 = agg.mean(table1.C1),
    ...                                       mean_c2 = agg.mean(table1.C2),
    ...                                       stats_c3 = agg.stats(table1.C3)))
    >>> print(t1_stats)

    Group by a field and aggregate to produce a new table:

    >>> table3 = (table1.group_by(table1.SEX)
    ...                 .aggregate(mean_height_data = agg.mean(table1.HT)))
    >>> table3.show()

    Join tables together inside an annotation expression:

    >>> table2 = table2.key_by('ID')
    >>> table1 = table1.annotate(B = table2[table1.ID].B)
    >>> table1.show()
    """

    @staticmethod
    def _from_java(jt):
        return Table(JavaTable(jt.tir()))

    def __init__(self, tir):
        super(Table, self).__init__()

        self._tir = tir
        self._jtir = tir.to_java_ir()
        self._jt = Env.hail().table.Table(Env.hc()._jhc, self._jtir)

        jttype = self._jtir.typ()

        self._row_axis = 'row'

        self._global_indices = Indices(axes=set(), source=self)
        self._row_indices = Indices(axes={self._row_axis}, source=self)

        self._global_type = hl.dtype(jttype.globalType().toString())
        self._row_type = hl.dtype(jttype.rowType().toString())

        assert isinstance(self._global_type, tstruct)
        assert isinstance(self._row_type, tstruct)

        self._globals = construct_reference('global', self._global_type, indices=self._global_indices)
        self._row = construct_reference('row', self._row_type, indices=self._row_indices)

        self._indices_from_ref = {'global': self._global_indices,
                                  'row': self._row_indices}

        self._key = hail.struct(
            **{k: self._row[k] for k in jiterable_to_list(jttype.key())})

        for k, v in itertools.chain(self._globals.items(),
                                    self._row.items()):
            self._set_field(k, v)


    def __getitem__(self, item):
        if isinstance(item, str):
            return self._get_field(item)
        else:
            try:
                return self.index(*wrap_to_tuple(item))
            except TypeError as e:
                raise TypeError(f"Table.__getitem__: invalid index argument(s)\n"
                                f"  Usage 1: field selection: ht['field']\n"
                                f"  Usage 2: Left distinct join: ht[ht2.key] or ht[ht2.field1, ht2.field2]") from e

    @property
    def key(self) -> StructExpression:
        """Row key struct.

        Examples
        --------

        List of key field names:

        >>> list(table1.key)
        ['ID']

        Number of key fields:

        >>> len(table1.key)
        1


        Returns
        -------
        :class:`.StructExpression`
        """
        return self._key

    def n_partitions(self):
        """Returns the number of partitions in the table.

        Returns
        -------
        :obj:`int`
        """
        return self._jt.nPartitions()

    def count(self):
        """Count the number of rows in the table.

        Examples
        --------

        >>> table1.count()
        4

        Returns
        -------
        :obj:`int`
        """
        return Env.hc()._backend.interpret(TableCount(self._tir))

    def _force_count(self):
        return self._jt.forceCount()

    @typecheck_method(caller=str,
                      row=expr_struct())
    def _select(self, caller, row):
        analyze(caller, row, self._row_indices)
        base, cleanup = self._process_joins(row)
        return cleanup(Table(TableMapRows(base._tir, row._ir)))

    @typecheck_method(caller=str, s=expr_struct())
    def _select_globals(self, caller, s):
        base, cleanup = self._process_joins(s)
        analyze(caller, s, self._global_indices)
        return cleanup(Table(TableMapGlobals(base._tir, s._ir)))

    @classmethod
    @typecheck_method(rows=anytype,
                      schema=nullable(hail_type),
                      key=table_key_type,
                      n_partitions=nullable(int))
    def parallelize(cls, rows, schema=None, key=None, n_partitions=None):
        """Parallelize a local array of structs into a distributed table.

        Examples
        --------
        Parallelize a list of dictionaries:

        >>> a = [ {'a': 5, 'b': 10}, {'a': 0, 'b': 200} ]
        >>> table = hl.Table.parallelize(hl.literal(a, 'array<struct{a: int, b: int}>'))
        >>> table.show()

        Warning
        -------
        Parallelizing very large local arrays will be slow.

        Parameters
        ----------
        rows
            List of row values, or expression of type ``array<struct{...}>``.
        schema : str or :class:`.HailType:`, optional
            Value type.
        key : Union[str, List[str]]], optional
            Key field(s).
        n_partitions : int, optional

        Returns
        -------
        :class:`.Table`
        """
        rows = to_expr(rows, dtype=hl.tarray(schema) if schema is not None else None)
        if not isinstance(rows.dtype.element_type, tstruct):
            raise TypeError("'parallelize' expects an array with element type 'struct', found '{}'"
                            .format(rows.dtype))
        table = Table(TableParallelize(rows._ir, n_partitions))
        if key is not None:
            table = table.key_by(*key)
        return table

    @typecheck_method(keys=oneof(str, expr_any),
                      named_keys=expr_any)
    def key_by(self, *keys, **named_keys) -> 'Table':
        """Key table by a new set of fields.

        Examples
        --------
        Assume `table1` is a :class:`.Table` with three fields: `C1`, `C2`
        and `C3`.

        Changing key fields:

        >>> table_result = table1.key_by('C2', 'C3')

        This keys the table by 'C2' and 'C3', preserving old keys as value fields.

        >>> table_result = table1.key_by(table1.C1)

        This keys the table by 'C1', preserving old keys as value fields.

        >>> table_result = table1.key_by(C1 = table1.C2, foo = table1.C1)

        This keys the table by fields named 'C1' and 'foo', which have values
        corresponding to the original 'C2' and 'C1' fields respectively. The original
        'C1' field has been overwritten by the new assignment, but the original
        'C2' field is preserved as a value field.

        Remove key:

        >>> table_result = table1.key_by()

        Notes
        -----
        This method is used to specify all the fields of a new row key. The old
        key fields may be overwritten by newly-assigned fields, as described in
        :meth:`.Table.annotate`. If not overwritten, they are preserved as non-key
        fields.

        See :meth:`.Table.select` for more information about how to define new
        key fields.

        Parameters
        ----------
        keys : varargs of type :obj:`str`
            Field(s) to key by.

        Returns
        -------
        :class:`.Table`
            Table with a new key.
        """
        key_fields = get_select_exprs("Table.key_by",
                                      keys, named_keys, self._row_indices,
                                      protect_keys=False)

        new_row = self.row.annotate(**key_fields)
        base, cleanup = self._process_joins(new_row)

        return cleanup(Table(
            TableKeyBy(
                TableMapRows(
                    TableKeyBy(base._tir, []),
                    new_row._ir),
                list(key_fields))))

    def annotate_globals(self, **named_exprs):
        """Add new global fields.

        Examples
        --------

        Add a new global field:

        >>> table_result = table1.annotate_globals(pops = ['EUR', 'AFR', 'EAS', 'SAS'])

        Note
        ----
        This method does not support aggregation.

        Parameters
        ----------
        named_exprs : varargs of :class:`.Expression`
            Annotation expressions.

        Returns
        -------
        :class:`.Table`
            Table with new global field(s).
        """
        named_exprs = {k: to_expr(v) for k, v in named_exprs.items()}
        for k, v in named_exprs.items():
            check_collisions(self._fields, k, self._global_indices)
        return self._select_globals('Table.annotate_globals', self.globals.annotate(**named_exprs))

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
        :class:`.Table`
            Table with specified global fields.
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
        return self._select_globals('Table.select_globals', hl.struct(**assignments))

    def transmute_globals(self, **named_exprs):
        """Similar to :meth:`.Table.annotate_globals`, but drops referenced fields.

        Notes
        -----
        This method adds new global fields according to `named_exprs`, and
        drops all global fields referenced in those expressions. See
        :meth:`.Table.transmute` for full documentation on how transmute
        methods work.

        See Also
        --------
        :meth:`.Table.transmute`, :meth:`.Table.select_globals`,
        :meth:`.Table.annotate_globals`

        Parameters
        ----------
        named_exprs : keyword args of :class:`.Expression`
            Annotation expressions.

        Returns
        -------
        :class:`.Table`
        """
        caller = 'Table.transmute_globals'
        e = get_annotate_exprs(caller, named_exprs, self._global_indices)
        fields_referenced = extract_refs_by_indices(e.values(), self._global_indices) - set(e.keys())

        return self._select_globals(caller,
                                    self.globals.annotate(**named_exprs).drop(*fields_referenced))


    def transmute(self, **named_exprs):
        """Add new fields and drop fields referenced.

        Examples
        --------

        Create a single field from an expression of `C1`, `C2`, and `C3`.

        >>> table4.show()
        +-------+------+---------+-------+-------+-------+-------+-------+
        |     A | B.B0 | B.B1    | C     | D.cat | D.dog |   E.A |   E.B |
        +-------+------+---------+-------+-------+-------+-------+-------+
        | int32 | bool | str     | bool  | int32 | int32 | int32 | int32 |
        +-------+------+---------+-------+-------+-------+-------+-------+
        |    32 | true | "hello" | false |     5 |     7 |     5 |     7 |
        +-------+------+---------+-------+-------+-------+-------+-------+

        >>> table_result = table4.transmute(F=table4.A + 2 * table4.E.B)
        >>> table_result.show()
        +------+---------+-------+-------+-------+-------+
        | B.B0 | B.B1    | C     | D.cat | D.dog |     F |
        +------+---------+-------+-------+-------+-------+
        | bool | str     | bool  | int32 | int32 | int32 |
        +------+---------+-------+-------+-------+-------+
        | true | "hello" | false |     5 |     7 |    46 |
        +------+---------+-------+-------+-------+-------+

        Notes
        -----
        This method functions to create new row-indexed fields and consume
        fields found in the expressions in `named_exprs`.

        All row-indexed top-level fields found in an expression are dropped
        after the new fields are created.

        Note
        ----
        :meth:`transmute` will not drop key fields.

        Warning
        -------
        References to fields inside a top-level struct will remove the entire
        struct, as field `E` was removed in the example above since `E.B` was
        referenced.

        Note
        ----
        This method does not support aggregation.

        Parameters
        ----------
        named_exprs : keyword args of :class:`.Expression`
            New field expressions.

        Returns
        -------
        :class:`.Table`
            Table with transmuted fields.
        """
        caller = "Table.transmute"
        e = get_annotate_exprs(caller, named_exprs, self._row_indices)
        fields_referenced = extract_refs_by_indices(e.values(), self._row_indices) - set(e.keys())
        fields_referenced -= set(self.key)

        return self._select(caller, self.row.annotate(**e).drop(*fields_referenced))

    def annotate(self, **named_exprs):
        """Add new fields.

        Examples
        --------

        Add field `Y` by computing the square of `X`:

        >>> table_result = table1.annotate(Y = table1.X ** 2)

        Add multiple fields simultaneously:

        >>> table_result = table1.annotate(A = table1.X / 2,
        ...                                B = table1.X + 21)

        Parameters
        ----------
        named_exprs : keyword args of :class:`.Expression`
            Expressions for new fields.

        Returns
        -------
        :class:`.Table`
            Table with new fields.
        """
        caller = "Table.annotate"
        e = get_annotate_exprs(caller, named_exprs, self._row_indices)
        return self._select(caller, self.row.annotate(**e))

    @typecheck_method(expr=expr_bool,
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
        expr : bool or :class:`.BooleanExpression`
            Filter expression.
        keep : bool
            Keep rows where `expr` is true.

        Returns
        -------
        :class:`.Table`
            Filtered table.
        """
        analyze('Table.filter', expr, self._row_indices)
        base, cleanup = self._process_joins(expr)

        return cleanup(Table(TableFilter(base._tir, filter_predicate_with_keep(expr._ir, keep))))

    @typecheck_method(exprs=oneof(Expression, str),
                      named_exprs=anytype)
    def select(self, *exprs, **named_exprs) -> 'Table':
        """Select existing fields or create new fields by name, dropping the rest.

        Examples
        --------
        Select a few old fields and compute a new one:

        >>> table_result = table1.select(table1.C1, Y=table1.Z - table1.X)

        Notes
        -----
        This method creates new row-indexed fields. If a created field shares its name
        with a global field of the table, the method will fail.

        Note
        ----

        **Using select**

        Select and its sibling methods (:meth:`.Table.select_globals`,
        :meth:`.MatrixTable.select_globals`, :meth:`.MatrixTable.select_rows`,
        :meth:`.MatrixTable.select_cols`, and :meth:`.MatrixTable.select_entries`) accept
        both variable-length (``f(x, y, z)``) and keyword (``f(a=x, b=y, c=z)``)
        arguments.

        Select methods will always preserve the key along that axis; e.g. for
        :meth:`.Table.select`, the table key will aways be kept. To modify the
        key, use :meth:`.key_by`.

        Variable-length arguments can be either strings or expressions that reference a
        (possibly nested) field of the table. Keyword arguments can be arbitrary
        expressions.

        **The following three usages are all equivalent**, producing a new table with
        fields `C1` and `C2` of `table1`, and the table key `ID`.

        First, variable-length string arguments:

        >>> table_result = table1.select('C1', 'C2')

        Second, field reference variable-length arguments:

        >>> table_result = table1.select(table1.C1, table1.C2)

        Last, expression keyword arguments:

        >>> table_result = table1.select(C1 = table1.C1, C2 = table1.C2)

        Additionally, the variable-length argument syntax also permits nested field
        references. Given the following struct field `s`:

        >>> table3 = table1.annotate(s = hl.struct(x=table1.X, z=table1.Z))

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
        exprs : variable-length args of :obj:`str` or :class:`.Expression`
            Arguments that specify field names or nested field reference expressions.
        named_exprs : keyword args of :class:`.Expression`
            Field names and the expressions to compute them.

        Returns
        -------
        :class:`.Table`
            Table with specified fields.
        """
        row_exprs = get_select_exprs('Table.select',
                                     exprs, named_exprs, self._row_indices,
                                     protect_keys=True)
        row = self.key.annotate(**row_exprs)

        return self._select('Table.select', row)

    @typecheck_method(exprs=oneof(str, Expression))
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
        exprs : varargs of :obj:`str` or :class:`.Expression`
            Names of fields to drop or field reference expressions.

        Returns
        -------
        :class:`.Table`
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
                assert isinstance(e, str)
                if e not in self._fields:
                    raise IndexError("table has no field '{}'".format(e))
                fields_to_drop.add(e)

        table = self
        if any(self._fields[field]._indices == self._global_indices for field in fields_to_drop):
            # need to drop globals
            new_global_fields = [f for f in table.globals if
                                 f not in fields_to_drop]
            table = table.select_globals(*new_global_fields)

        if any(self._fields[field]._indices == self._row_indices for field in fields_to_drop):
            # need to drop row fields
            for f in fields_to_drop:
                check_keys(f, self._row_indices)
            row_fields = set(table.row)
            to_drop = [f for f in fields_to_drop if f in row_fields]
            table = table._select('drop', table.row.drop(*to_drop))

        return table

    @typecheck_method(output=str,
               types_file=nullable(str),
               header=bool,
               parallel=nullable(enumeration('separate_header', 'header_per_shard')))
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
        output : :obj:`str`
            URI at which to write exported file.
        types_file : :obj:`str`, optional
            URI at which to write file containing field type information.
        header : :obj:`bool`
            Include a header in the file.
        parallel : :obj:`str`, optional
            If None, a single file is produced, otherwise a
            folder of file shards is produced. If 'separate_header',
            the header file is output separately from the file shards. If
            'header_per_shard', each file shard has a header. If set to None
            the export will be slower.
        """

        self._jt.export(output, types_file, header, Env.hail().utils.ExportType.getExportType(parallel))

    def group_by(self, *exprs, **named_exprs) -> 'GroupedTable':
        """Group by a new key for use with :meth:`.GroupedTable.aggregate`.

        Examples
        --------
        Compute the mean value of `X` and the sum of `Z` per unique `ID`:

        >>> table_result = (table1.group_by(table1.ID)
        ...                       .aggregate(meanX = agg.mean(table1.X), sumZ = agg.sum(table1.Z)))

        Group by a height bin and compute sex ratio per bin:

        >>> table_result = (table1.group_by(height_bin = table1.HT // 20)
        ...                       .aggregate(fraction_female = agg.fraction(table1.SEX == 'F')))

        Notes
        -----
        This function is always followed by :meth:`.GroupedTable.aggregate`. Follow the
        link for documentation on the aggregation step.

        Note
        ----
        **Using group_by**

        **group_by** and its sibling methods (:meth:`.MatrixTable.group_rows_by` and
        :meth:`.MatrixTable.group_cols_by`) accept both variable-length (``f(x, y, z)``)
        and keyword (``f(a=x, b=y, c=z)``) arguments.

        Variable-length arguments can be either strings or expressions that reference a
        (possibly nested) field of the table. Keyword arguments can be arbitrary
        expressions.

        **The following three usages are all equivalent**, producing a
        :class:`.GroupedTable` grouped by fields `C1` and `C2` of `table1`.

        First, variable-length string arguments:

        >>> table_result = (table1.group_by('C1', 'C2')
        ...                       .aggregate(meanX = agg.mean(table1.X)))

        Second, field reference variable-length arguments:

        >>> table_result = (table1.group_by(table1.C1, table1.C2)
        ...                       .aggregate(meanX = agg.mean(table1.X)))

        Last, expression keyword arguments:

        >>> table_result = (table1.group_by(C1 = table1.C1, C2 = table1.C2)
        ...                       .aggregate(meanX = agg.mean(table1.X)))

        Additionally, the variable-length argument syntax also permits nested field
        references. Given the following struct field `s`:

        >>> table3 = table1.annotate(s = hl.struct(x=table1.X, z=table1.Z))

        The following two usages are equivalent, grouping by one field, `x`:

        >>> table_result = (table3.group_by(table3.s.x)
        ...                       .aggregate(meanX = agg.mean(table3.X)))

        >>> table_result = (table3.group_by(x = table3.s.x)
        ...                       .aggregate(meanX = agg.mean(table3.X)))

        The keyword argument syntax permits arbitrary expressions:

        >>> table_result = (table1.group_by(foo=table1.X ** 2 + 1)
        ...                       .aggregate(meanZ = agg.mean(table1.Z)))

        These syntaxes can be mixed together, with the stipulation that all keyword arguments
        must come at the end due to Python language restrictions.

        >>> table_result = (table1.group_by(table1.C1, 'C2', height_bin = table1.HT // 20)
        ...                       .aggregate(meanX = agg.mean(table1.X)))

        Note
        ----
        This method does not support aggregation in key expressions.

        Arguments
        ---------
        exprs : varargs of type str or :class:`.Expression`
            Field names or field reference expressions.
        named_exprs : keyword args of type :class:`.Expression`
            Field names and expressions to compute them.

        Returns
        -------
        :class:`.GroupedTable`
            Grouped table; use :meth:`.GroupedTable.aggregate` to complete the aggregation.
        """
        groups = []
        for e in exprs:
            if isinstance(e, str):
                e = self[e]
            else:
                e = to_expr(e)
            analyze('Table.group_by', e, self._row_indices)
            if not e._ir.is_nested_field:
                raise ExpressionException("method 'group_by' expects keyword arguments for complex expressions")
            key = e._ir.name
            groups.append((key, e))
        for k, e in named_exprs.items():
            e = to_expr(e)
            analyze('Table.group_by', e, self._row_indices)
            groups.append((k, e))

        return GroupedTable(self, groups)

    def aggregate(self, expr):
        """Aggregate over rows into a local value.

        Examples
        --------
        Aggregate over rows:

        >>> table1.aggregate(hl.struct(fraction_male=agg.fraction(table1.SEX == 'M'),
        ...                            mean_x=agg.mean(table1.X)))
        Struct(fraction_male=0.5, mean_x=6.5)

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
        expr = to_expr(expr)
        base, _ = self._process_joins(expr)
        analyze('Table.aggregate', expr, self._global_indices, {self._row_axis})

        return Env.hc()._backend.interpret(TableAggregate(base._tir, expr._ir))

    @typecheck_method(output=str,
                      overwrite=bool,
                      stage_locally=bool,
                      _codec_spec=nullable(str))
    def write(self, output: str, overwrite = False, stage_locally: bool = False,
              _codec_spec: Optional[str] = None):
        """Write to disk.

        Examples
        --------

        >>> table1.write('output/table1.ht')

        Warning
        -------
        Do not write to a path that is being read from in the same computation.

        Parameters
        ----------
        output : str
            Path at which to write.
        stage_locally: bool
            If ``True``, major output will be written to temporary local storage
            before being copied to ``output``.
        overwrite : bool
            If ``True``, overwrite an existing file at the destination.
        """

        Env.hc()._backend.interpret(TableWrite(self._tir, output, overwrite, stage_locally, _codec_spec))

    @typecheck_method(n=int, width=int, truncate=nullable(int), types=bool, handler=anyfunc)
    def show(self, n=10, width=90, truncate=None, types=True, handler=print):
        """Print the first few rows of the table to the console.

        Examples
        --------
        Show the first lines of the table:

        >>> table1.show()
        +-------+-------+-----+-------+-------+-------+-------+-------+
        |    ID |    HT | SEX |     X |     Z |    C1 |    C2 |    C3 |
        +-------+-------+-----+-------+-------+-------+-------+-------+
        | int32 | int32 | str | int32 | int32 | int32 | int32 | int32 |
        +-------+-------+-----+-------+-------+-------+-------+-------+
        |     1 |    65 | "M" |     5 |     4 |     2 |    50 |     5 |
        |     2 |    72 | "M" |     6 |     3 |     2 |    61 |     1 |
        |     3 |    70 | "F" |     7 |     3 |    10 |    81 |    -5 |
        |     4 |    60 | "F" |     8 |     2 |    11 |    90 |   -10 |
        +-------+-------+-----+-------+-------+-------+-------+-------+

        Parameters
        ----------
        n : :obj:`int`
            Maximum number of rows to show.
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
        handler(self._show(n,width, truncate, types))

    def _show(self, n=10, width=90, truncate=None, types=True):
        return self._jt.showString(n, joption(truncate), types, width)

    def index(self, *exprs):
        """Expose the row values as if looked up in a dictionary, indexing
        with `exprs`.

        Examples
        --------
        In the example below, both `table1` and `table2` are keyed by one
        field `ID` of type ``int``.

        >>> table_result = table1.select(B = table2.index(table1.ID).B)
        >>> table_result.B.show()
        +-------+----------+
        |    ID | B        |
        +-------+----------+
        | int32 | str      |
        +-------+----------+
        |     1 | "cat"    |
        |     2 | "dog"    |
        |     3 | "mouse"  |
        |     4 | "rabbit" |
        +-------+----------+

        Using `key` as the sole index expression is equivalent to passing all
        key fields individually:

        >>> table_result = table1.select(B = table2.index(table1.key).B)

        It is also possible to use non-key fields or expressions as the index
        expressions:

        >>> table_result = table1.select(B = table2.index(table1.C1 % 4).B)
        >>> table_result.show()
        +-------+---------+
        |    ID | B       |
        +-------+---------+
        | int32 | str     |
        +-------+---------+
        |     1 | "dog"   |
        |     2 | "dog"   |
        |     3 | "dog"   |
        |     4 | "mouse" |
        +-------+---------+

        Notes
        -----
        :meth:`.Table.index` is used to expose one table's fields for use in
        expressions involving the another table or matrix table's fields. The
        result of the method call is a struct expression that is usable in the
        same scope as `exprs`, just as if `exprs` were used to look up values of
        the table in a dictionary.

        The type of the struct expression is the same as the indexed table's
        :meth:`.row_value` (the key fields are removed, as they are available
        in the form of the index expressions).

        Note
        ----
        There is a shorthand syntax for :meth:`.Table.index` using square
        brackets (the Python ``__getitem__`` syntax). This syntax is preferred.

        >>> table_result = table1.select(B = table2[table1.ID].B)

        Parameters
        ----------
        exprs : variable-length args of :class:`.Expression`
            Index expressions.

        Returns
        -------
        :class:`.StructExpression`
        """
        exprs = tuple(exprs)
        if not len(exprs) > 0:
            raise ValueError('Require at least one expression to index')
        non_exprs = list(filter(lambda e: not isinstance(e, Expression), exprs))
        if non_exprs:
            raise TypeError(f"'Table.index': arguments must be expressions, found {non_exprs}")

        from hail.matrixtable import MatrixTable
        indices, aggregations = unify_all(*exprs)
        src = indices.source

        if src is None or len(indices.axes) == 0:
            # FIXME: this should be OK: table[m.global_index_into_table]
            raise ExpressionException('Cannot index table with a scalar expression')

        def types_compatible(left, right):
            left = list(left)
            right = list(right)
            return (types_match(left, right)
                    or (len(left) == 1
                        and len(right) == 1
                        and isinstance(left[0].dtype, tinterval)
                        and left[0].dtype.point_type == right[0].dtype))

        if not types_compatible(self.key.values(), exprs):
            if (len(exprs) == 1
                    and isinstance(exprs[0], TupleExpression)
                    and types_compatible(self.key.values(), exprs[0])):
                return self.index(*exprs[0])
            elif (len(exprs) == 1
                  and isinstance(exprs[0], StructExpression)
                  and types_compatible(self.key.values(), exprs[0].values())):
                return self.index(*exprs[0].values())
            elif len(exprs) != len(self.key):
                raise ExpressionException(f'Key mismatch: table has {len(self.key)} key fields, '
                                          f'found {len(exprs)} index expressions.')
            else:
                raise ExpressionException(f"Key type mismatch: cannot index table with given expressions:\n"
                                          f"  Table key:         {', '.join(str(t) for t in self.key.dtype.values())}\n"
                                          f"  Index Expressions: {', '.join(str(e.dtype) for e in exprs)}")

        uid = Env.get_uid()

        new_schema = self.row_value.dtype

        if isinstance(src, Table):
            for e in exprs:
                analyze('Table.index', e, src._row_indices)

            is_key = len(exprs) == len(src.key) and all(
                expr is key_field for expr, key_field in zip(exprs, src.key.values()))
            is_interval = (len(self.key) == 1
                           and isinstance(self.key[0].dtype, hl.tinterval)
                           and exprs[0].dtype == self.key[0].dtype.point_type)

            if not is_key:
                uids = [Env.get_uid() for i in range(len(exprs))]
                all_uids = uids[:]
            else:
                all_uids = []

            def joiner(left):
                if not is_key:
                    original_key = list(left.key)
                    left = Table._from_java(left.key_by()._jt.mapRows(str(Apply('annotate',
                                                             left._row._ir,
                                                             hl.struct(**dict(zip(uids, exprs)))._ir)))
                                 ).key_by(*uids)
                    rekey_f = lambda t: t.key_by(*original_key)
                else:
                    rekey_f = identity

                if is_interval:
                    left = Table._from_java(left._jt.intervalJoin(self._jt, uid))
                else:
                    left = Table._from_java(left._jt.leftJoinRightDistinct(self._jt, uid))
                return rekey_f(left)

            all_uids.append(uid)
            ir = Join(GetField(TopLevelReference('row'), uid),
                      all_uids,
                      exprs,
                      joiner)
            return construct_expr(ir, new_schema, indices, aggregations)
        elif isinstance(src, MatrixTable):
            for e in exprs:
                analyze('Table.index', e, src._entry_indices)

            right = self
            # match on indices to determine join type
            if indices == src._entry_indices:
                raise NotImplementedError('entry-based matrix joins')
            elif indices == src._row_indices:
                is_subset_row_key = len(exprs) <= len(src.row_key) and all(
                    expr is key_field for expr, key_field in zip(exprs, src.row_key.values()))
                is_interval = (len(self.key) == 1
                               and isinstance(self.key[0].dtype, hl.tinterval)
                               and exprs[0].dtype == self.key[0].dtype.point_type)

                if is_interval and (len(exprs) != 1 or
                                    exprs[0] is not src.row_key[0] or
                                    self.key[0].dtype.point_type != exprs[0].dtype):
                    raise ExpressionException(f"Key type mismatch: cannot index table with given expressions:\n"
                                              f"  Table key:         {', '.join(str(t) for t in self.key.dtype.values())}\n"
                                              f"  Index Expressions: {', '.join(str(e.dtype) for e in exprs)}")

                if is_subset_row_key or is_interval:
                    key = None
                else:
                    key = [str(k._ir) for k in exprs]
                joiner = lambda left: MatrixTable._from_java(left._jmt.annotateRowsTableIR(right._jt, uid, key))
                ast = Join(GetField(TopLevelReference('va'), uid),
                           [uid],
                           exprs,
                           joiner)
                return construct_expr(ast, new_schema, indices, aggregations)
            elif indices == src._col_indices:
                all_uids = [uid]
                if len(exprs) == len(src.col_key) and all([
                        exprs[i] is src.col_key[i] for i in range(len(exprs))]):
                    # key is already correct
                    def joiner(left):
                        return MatrixTable._from_java(left._jmt.annotateColsTable(right._jt, uid))
                else:
                    index_uid = Env.get_uid()
                    uids = [Env.get_uid() for _ in exprs]

                    all_uids.append(index_uid)
                    all_uids.extend(uids)
                    def joiner(left: MatrixTable):
                        prev_key = list(src.col_key)
                        joined = (src
                                  .annotate_cols(**dict(zip(uids, exprs)))
                                  .add_col_index(index_uid)
                                  .key_cols_by(*uids)
                                  .cols()
                                  .select(index_uid)
                                  .join(self, 'inner')
                                  .key_by(index_uid)
                                  .drop(*uids))
                        result = MatrixTable._from_java(left.add_col_index(index_uid)
                                                        .key_cols_by(index_uid)
                                                        ._jmt
                                                        .annotateColsTable(joined._jt, uid)).key_cols_by(*prev_key)
                        return result
                ir = Join(GetField(TopLevelReference('sa'), uid),
                          all_uids,
                          exprs,
                          joiner)
                return construct_expr(ir, new_schema, indices, aggregations)
            else:
                raise NotImplementedError()
        else:
            raise TypeError("Cannot join with expressions derived from '{}'".format(src.__class__))

    def index_globals(self):
        """Return this table's global variables for use in another
        expression context.

        Examples
        --------
        >>> table_result = table2.annotate(C = table2.A * table1.index_globals().global_field_1)

        Returns
        -------
        :class:`.StructExpression`
        """
        uid = Env.get_uid()

        def joiner(obj):
            from hail.matrixtable import MatrixTable
            if isinstance(obj, MatrixTable):
                return MatrixTable._from_java(Env.jutils().joinGlobals(obj._jmt, self._jt, uid))
            assert isinstance(obj, Table)
            return Table._from_java(Env.jutils().joinGlobals(obj._jt, self._jt, uid))

        ir = Join(GetField(TopLevelReference('global'), uid),
                  [uid],
                  [],
                  joiner)
        return construct_expr(ir, self.globals.dtype)

    def _process_joins(self, *exprs):
        return process_joins(self, exprs)

    def cache(self):
        """Persist this table in memory.

        Examples
        --------
        Persist the table in memory:

        >>> table = table.cache() # doctest: +SKIP

        Notes
        -----

        This method is an alias for :func:`persist("MEMORY_ONLY") <hail.Table.persist>`.

        Returns
        -------
        :class:`.Table`
            Cached table.
        """
        return self.persist('MEMORY_ONLY')

    @typecheck_method(storage_level=storage_level)
    def persist(self, storage_level='MEMORY_AND_DISK'):
        """Persist this table in memory or on disk.

        Examples
        --------
        Persist the table to both memory and disk:

        >>> table = table.persist() # doctest: +SKIP

        Notes
        -----

        The :meth:`.Table.persist` and :meth:`.Table.cache` methods store the
        current table on disk or in memory temporarily to avoid redundant computation
        and improve the performance of Hail pipelines. This method is not a substitution
        for :meth:`.Table.write`, which stores a permanent file.

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
        :class:`.Table`
            Persisted table.
        """
        return Table._from_java(self._jt.persist(storage_level))

    def unpersist(self):
        """
        Unpersists this table from memory/disk.

        Notes
        -----
        This function will have no effect on a table that was not previously
        persisted.

        Returns
        -------
        :class:`.Table`
            Unpersisted table.
        """
        return Table._from_java(self._jt.unpersist())

    def collect(self):
        """Collect the rows of the table into a local list.

        Examples
        --------
        Collect a list of all `X` records:

        >>> all_xs = [row['X'] for row in table1.select(table1.X).collect()]

        Notes
        -----
        This method returns a list whose elements are of type :class:`.Struct`. Fields
        of these structs can be accessed similarly to fields on a table, using dot
        methods (``struct.foo``) or string indexing (``struct['foo']``).

        Warning
        -------
        Using this method can cause out of memory errors. Only collect small tables.

        Returns
        -------
        :obj:`list` of :class:`.Struct`
            List of rows.
        """
        return hl.tarray(self.row.dtype)._from_json(self._jt.collectJSON())

    def describe(self, handler=print):
        """Print information about the fields in the table."""

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

        row_key = '[' + ', '.join("'{name}'".format(name=f) for f in self.key) + ']'

        s = '----------------------------------------\n' \
            'Global fields:{g}\n' \
            '----------------------------------------\n' \
            'Row fields:{r}\n' \
            '----------------------------------------\n' \
            'Key: {rk}\n' \
            '----------------------------------------'.format(g=global_fields,
                                                              rk=row_key,
                                                              r=row_fields)
        handler(s)

    @typecheck_method(name=str)
    def add_index(self, name='idx'):
        """Add the integer index of each row as a new row field.

        Examples
        --------

        >>> table_result = table1.add_index()
        >>> table_result.show()  # doctest: +NOTEST
        +-------+-------+-----+-------+-------+-------+-------+-------+-------+
        |    ID |    HT | SEX |     X |     Z |    C1 |    C2 |    C3 |   idx |
        +-------+-------+-----+-------+-------+-------+-------+-------+-------+
        | int32 | int32 | str | int32 | int32 | int32 | int32 | int32 | int64 |
        +-------+-------+-----+-------+-------+-------+-------+-------+-------+
        |     1 |    65 | M   |     5 |     4 |     2 |    50 |     5 |     0 |
        |     2 |    72 | M   |     6 |     3 |     2 |    61 |     1 |     1 |
        |     3 |    70 | F   |     7 |     3 |    10 |    81 |    -5 |     2 |
        |     4 |    60 | F   |     8 |     2 |    11 |    90 |   -10 |     3 |
        +-------+-------+-----+-------+-------+-------+-------+-------+-------+

        Notes
        -----

        This method returns a table with a new field whose name is given by
        the `name` parameter, with type :py:data:`.tint64`. The value of this field
        is the integer index of each row, starting from 0. Methods that respect
        ordering (like :meth:`.Table.take` or :meth:`.Table.export`) will
        return rows in order.

        This method is also helpful for creating a unique integer index for
        rows of a table so that more complex types can be encoded as a simple
        number for performance reasons.

        Parameters
        ----------
        name : str
            Name of index field.

        Returns
        -------
        :class:`.Table`
            Table with a new index field.
        """

        return self.annotate(**{name: hl.scan.count()})

    @typecheck_method(tables=table_type)
    def union(self, *tables):
        """Union the rows of multiple tables.

        Examples
        --------

        Take the union of rows from two tables:

        >>> union_table = table1.union(other_table)

        Notes
        -----

        If a row appears in both tables identically, it is duplicated in the
        result. The left and right tables must have the same schema and key.

        Parameters
        ----------
        tables : varargs of :class:`.Table`
            Tables to union.

        Returns
        -------
        :class:`.Table`
            Table with all rows from each component table.
        """
        left_key = list(self.key)
        for i, ht, in enumerate(tables):
            right_key = list(ht.key)
            if not ht.row.dtype == self.row.dtype:
                raise ValueError(f"'union': table {i} has a different row type.\n"
                                f"  Expected:  {self.row.dtype}\n"
                                f"  Table {i}: {ht.row.dtype}")
            elif left_key != right_key:
                raise ValueError(f"'union': table {i} has a different key."
                                f"  Expected:  {left_key}\n"
                                f"  Table {i}: {right_key}")
        return Table(TableUnion([self._tir] + [table._tir for table in tables]))

    @typecheck_method(n=int)
    def take(self, n):
        """Collect the first `n` rows of the table into a local list.

        Examples
        --------
        Take the first three rows:

        >>> first3 = table1.take(3)
        >>> first3
        [Struct(ID=1, HT=65, SEX='M', X=5, Z=4, C1=2, C2=50, C3=5),
         Struct(ID=2, HT=72, SEX='M', X=6, Z=3, C1=2, C2=61, C3=1),
         Struct(ID=3, HT=70, SEX='F', X=7, Z=3, C1=10, C2=81, C3=-5)]

        Notes
        -----

        This method does not need to look at all the data in the table, and
        allows for fast queries of the start of the table.

        This method is equivalent to :meth:`.Table.head` followed by
        :meth:`.Table.collect`.

        Parameters
        ----------
        n : int
            Number of rows to take.

        Returns
        -------
        :obj:`list` of :class:`.Struct`
            List of row structs.
        """

        return self.head(n).collect()

    @typecheck_method(n=int)
    def head(self, n):
        """Subset table to first `n` rows.

        Examples
        --------
        Subset to the first three rows:

        >>> table_result = table1.head(3)
        >>> table_result.count()
        3

        Notes
        -----

        The number of partitions in the new table is equal to the number of
        partitions containing the first `n` rows.

        Parameters
        ----------
        n : int
            Number of rows to include.

        Returns
        -------
        :class:`.Table`
            Table including the first `n` rows.
        """

        return Table(TableHead(self._tir, n))

    @typecheck_method(p=numeric,
                      seed=nullable(int))
    def sample(self, p, seed=None):
        """Downsample the table by keeping each row with probability ``p``.

        Examples
        --------

        Downsample the table to approximately 1% of its rows.

        >>> small_table1 = table1.sample(0.01)

        Parameters
        ----------
        p : :obj:`float`
            Probability of keeping each row.
        seed : :obj:`int`
            Random seed.

        Returns
        -------
        :class:`.Table`
            Table with approximately ``p * n_rows`` rows.
        """

        if not (0 <= p <= 1):
            raise ValueError("Requires 'p' in [0,1]. Found p={}".format(p))

        return self.filter(hl.rand_bool(p, seed))

    @typecheck_method(n=int,
                      shuffle=bool)
    def repartition(self, n, shuffle=True):
        """Change the number of distributed partitions.

        Examples
        --------
        Repartition to 10 partitions:

        >>> table_result = table1.repartition(10)

        Warning
        -------
        When `shuffle` is ``False``, `repartition` can only decrease the number
        of partitions and simply combines adjacent partitions to achieve the
        desired number. It does not attempt to rebalance and so can produce a
        heavily unbalanced dataset. An unbalanced dataset can be inefficient to
        operate on because the work is not evenly distributed across partitions.

        Parameters
        ----------
        n : int
            Desired number of partitions.
        shuffle : bool
            If ``True``, shuffle data. Otherwise, naively coalesce.

        Returns
        -------
        :class:`.Table`
            Repartitioned table.
        """

        return Table(TableRepartition(self._tir, n, shuffle))

    @typecheck_method(right=table_type,
                      how=enumeration('inner', 'outer', 'left', 'right'),
                      _mangle=anyfunc)
    def join(self,
             right: 'Table',
             how='inner',
             _mangle: Callable[[str, int], str] = lambda s, i: f'{s}_{i}') -> 'Table':
        """Join two tables together.

        Examples
        --------
        Join `table1` to `table2` to produce `table_joined`:

        >>> table_joined = table1.key_by('ID').join(table2.key_by('ID'))

        Notes
        -----
        Hail supports four types of joins specified by `how`:

        - **inner** -- Key must be present in both the left and right tables.
        - **outer** -- Key present in either the left or the right. For keys
          only in the left table, the right table's fields will be missing.
          For keys only in the right table, the left table's fields will be
          missing.
        - **left** -- Key present in the left table. For keys not found on
          the right, the right table's fields will be missing.
        - **right** -- Key present in the right table. For keys not found on
          the right, the right table's fields will be missing.

        Both tables must have the same number of keys and the corresponding
        types of each key must be the same (order matters), but the key names
        can be different. For example, if `table1` is keyed by fields ``['a',
        'b']``, both of type ``int32``, and `table2` is keyed by fields ``['c',
        'd']``, both of type ``int32``, then the two tables can be joined (their
        rows will be joined where ``table1.a == table2.c`` and ``table1.b ==
        table2.d``).

        The key fields and order from the left table are preserved,
        while the key fields from the right table are not present in
        the result.

        Note
        ----
        These join methods implement a traditional `Cartesian product
        <https://en.wikipedia.org/wiki/Cartesian_product>`__ join, and
        the number of records in the resulting table can be larger than
        the number of records on the left or right if duplicate keys are
        present.

        Parameters
        ----------
        right : :class:`.Table`
            Table with which to join.
        how : :obj:`str`
            Join type. One of "inner", "outer", "left", "right".

        Returns
        -------
        :class:`.Table`
            Joined table.

        """
        left_key_types = list(self.key.dtype.values())
        right_key_types = list(right.key.dtype.values())
        if not left_key_types == right_key_types:
            raise ValueError(f"'join': key mismatch:\n  "
                             f"  left:  [{', '.join(str(t) for t in left_key_types)}]\n  "
                             f"  right: [{', '.join(str(t) for t in right_key_types)}]")
        seen = set(self._fields.keys())

        renames = {}

        for field in right._fields:
            if field in seen:
                i = 1
                while i < 100:
                    mod = _mangle(field, i)
                    if mod not in seen:
                        renames[field] = mod
                        seen.add(mod)
                        break
                    i += 1
                else:
                    raise RecursionError(f'cannot rename field {repr(field)} after 99'
                                         f' mangling attempts')
            else:
                seen.add(field)

        if renames:
            right = right.rename(renames)
            info(f'Table.join: renamed the following fields on the right to avoid name conflicts:' +
                 ''.join(f'\n    {repr(k)} -> {repr(v)}' for k, v in renames.items()))

        return Table(TableJoin(self._tir, right._tir, how, len(self.key)))

    @typecheck_method(expr=BooleanExpression)
    def all(self, expr):
        """Evaluate whether a boolean expression is true for all rows.

        Examples
        --------
        Test whether `C1` is greater than 5 in all rows of the table:

        >>> if table1.all(table1.C1 == 5):
        ...     print("All rows have C1 equal 5.")

        Parameters
        ----------
        expr : :class:`.BooleanExpression`
            Expression to test.

        Returns
        -------
        :obj:`bool`
        """
        return self.aggregate(hail.agg.all(expr))

    @typecheck_method(expr=BooleanExpression)
    def any(self, expr):
        """Evaluate whether a Boolean expression is true for at least one row.

        Examples
        --------

        Test whether `C1` is equal to 5 any row in any row of the table:

        >>> if table1.any(table1.C1 == 5):
        ...     print("At least one row has C1 equal 5.")

        Parameters
        ----------
        expr : :class:`.BooleanExpression`
            Boolean expression.

        Returns
        -------
        :obj:`bool`
            ``True`` if the predicate evaluated for ``True`` for any row, otherwise ``False``.
        """
        return self.aggregate(hail.agg.any(expr))

    @typecheck_method(mapping=dictof(str, str))
    def rename(self, mapping):
        """Rename fields of the table.

        Examples
        --------
        Rename `C1` to `col1` and `C2` to `col2`:

        >>> table_result = table1.rename({'C1' : 'col1', 'C2' : 'col2'})

        Parameters
        ----------
        mapping : :obj:`dict` of :obj:`str`, :obj:`str`
            Mapping from old field names to new field names.

        Notes
        -----
        Any field that does not appear as a key in `mapping` will not be
        renamed.

        Returns
        -------
        :class:`.Table`
            Table with renamed fields.
        """
        seen = {}

        row_map = {}
        global_map = {}

        for k, v in mapping.items():
            if v in seen:
                raise ValueError(
                    "Cannot rename two fields to the same name: attempted to rename {} and {} both to {}".format(
                        repr(seen[v]), repr(k), repr(v)))
            if v in self._fields and v not in mapping:
                raise ValueError("Cannot rename {} to {}: field already exists.".format(repr(k), repr(v)))
            seen[v] = k
            if self[k]._indices == self._row_indices:
                row_map[k] = v
            else:
                assert self[k]._indices == self._global_indices
                global_map[k] = v

        stray = set(mapping.keys()) - set(seen.values())
        if stray:
            raise ValueError(f"found rename rules for fields not present in table: {list(stray)}")

        return Table(TableRename(self._tir, row_map, global_map))

    def expand_types(self):
        """Expand complex types into structs and arrays.

        Examples
        --------

        >>> table_result = table1.expand_types()

        Notes
        -----
        Expands the following types: :class:`.tlocus`, :class:`.tinterval`,
        :class:`.tset`, :class:`.tdict`, :class:`.ttuple`.

        The only types that will remain after this method are:
        :py:data:`.tbool`, :py:data:`.tint32`, :py:data:`.tint64`,
        :py:data:`.tfloat64`, :py:data:`.tfloat32`, :class:`.tarray`,
        :class:`.tstruct`.

        Note, expand_types always returns an unkeyed table.

        Returns
        -------
        :class:`.Table`
            Expanded table.
        """

        if len(self.key) == 0:
            t = self
        else:
            t = self.key_by()
        return Table._from_java(t._jt.expandTypes())

    def flatten(self):
        """Flatten nested structs.

        Examples
        --------
        Flatten table:

        >>> table_result = table1.flatten()

        Notes
        -----
        Consider a table with signature

        .. code-block:: text

            a: struct{
                p: int32,
                q: str
            },
            b: int32,
            c: struct{
                x: str,
                y: array<struct{
                    y: str,
                    z: str
                }>
            }

        and key ``a``.  The result of flatten is

        .. code-block:: text

            a.p: int32
            a.q: str
            b: int32
            c.x: str
            c.y: array<struct{
                y: str,
                z: str
            }>

        with key ``a.p, a.q``.

        Note, structures inside collections like arrays or sets will not be
        flattened.

        Note, the result of flatten is always unkeyed.

        Warning
        -------
        Flattening a table will produces fields that cannot be referenced using
        the ``table.<field>`` syntax, e.g. "a.b". Reference these fields using
        square bracket lookups: ``table['a.b']``.

        Returns
        -------
        :class:`.Table`
            Table with a flat schema (no struct fields).
        """

        if len(self.key) == 0:
            t = self
        else:
            t = self.key_by()
        return Table._from_java(t._jt.flatten())

    @typecheck_method(exprs=oneof(str, Expression, Ascending, Descending))
    def order_by(self, *exprs):
        """Sort by the specified fields. Unkeys the table, if keyed.

        Examples
        --------
        Four equivalent ways to order the table by field `HT`, ascending:

        >>> sorted_table = table1.order_by(table1.HT)

        >>> sorted_table = table1.order_by('HT')

        >>> sorted_table = table1.order_by(hl.asc(table1.HT))

        >>> sorted_table = table1.order_by(hl.asc('HT'))

        Notes
        -----
        Missing values are sorted after non-missing values. When multiple
        fields are passed, the table will be sorted first by the first
        argument, then the second, etc.

        Note
        ----
        This method unkeys the table.

        Parameters
        ----------
        exprs : varargs of :class:`.Ascending` or :class:`.Descending` or :class:`.Expression` or :obj:`str`
            Fields to sort by.

        Returns
        -------
        :class:`.Table`
            Table sorted by the given fields.
        """
        sort_fields = []
        for e in exprs:
            if isinstance(e, str):
                expr = self[e]
                if not expr._indices == self._row_indices:
                    raise ValueError("Sort fields must be row-indexed, found global field '{}'".format(e))
                sort_fields.append((e, 'A'))
            elif isinstance(e, Expression):
                if not e in self._fields_inverse:
                    raise ValueError("Expect top-level field, found a complex expression")
                if not e._indices == self._row_indices:
                    raise ValueError("Sort fields must be row-indexed, found global field '{}'".format(e))
                sort_fields.append((self._fields_inverse[e], 'A'))
            else:
                assert isinstance(e, Ascending) or isinstance(e, Descending)
                sort_type = 'A' if isinstance(e, Ascending) else 'D'
                if isinstance(e.col, str):
                    expr = self[e.col]
                    if not expr._indices == self._row_indices:
                        raise ValueError("Sort fields must be row-indexed, found global field '{}'".format(e))
                    sort_fields.append((e.col, sort_type))
                else:
                    if not e.col in self._fields_inverse:
                        raise ValueError("Expect top-level field, found a complex expression")
                    if not e.col._indices == self._row_indices:
                        raise ValueError("Sort fields must be row-indexed, found global field '{}'".format(e))
                    e.col = self._fields_inverse[e.col]
                    sort_fields.append((e.col, sort_type))
        return Table(TableOrderBy(self.key_by()._tir, sort_fields))

    @typecheck_method(field=oneof(str, Expression),
                      name=nullable(str))
    def explode(self, field, name=None):
        """Explode rows along a top-level field of the table.

        Each row is copied for each element of `field`.
        The explode operation unpacks the elements in a field of type
        ``Array`` or ``Set`` into its own row. If an empty ``Array`` or ``Set``
        is exploded, the entire row is removed from the table.

        Examples
        --------

        `people_table` is a :class:`.Table` with three fields: `Name`, `Age` and `Children`.

        >>> people_table.show()
        +------------+-------+--------------------------+
        | Name       |   Age | Children                 |
        +------------+-------+--------------------------+
        | str        | int32 | array<str>               |
        +------------+-------+--------------------------+
        | "Alice"    |    34 | ["Dave","Ernie","Frank"] |
        | "Bob"      |    51 | ["Gaby","Helen"]         |
        | "Caroline" |    10 | []                       |
        +------------+-------+--------------------------+

        :meth:`.Table.explode` can be used to produce a distinct row for each
        element in the `Children` field:

        >>> exploded = people_table.explode('Children')
        >>> exploded.show()
        +---------+-------+----------+
        | Name    |   Age | Children |
        +---------+-------+----------+
        | str     | int32 | str      |
        +---------+-------+----------+
        | "Alice" |    34 | "Dave"   |
        | "Alice" |    34 | "Ernie"  |
        | "Alice" |    34 | "Frank"  |
        | "Bob"   |    51 | "Gaby"   |
        | "Bob"   |    51 | "Helen"  |
        +---------+-------+----------+

        The `name` parameter can be used to produce more appropriate field
        names:

        >>> exploded = people_table.explode('Children', name='Child')
        >>> exploded.show()
        +---------+-------+---------+
        | Name    |   Age | Child   |
        +---------+-------+---------+
        | str     | int32 | str     |
        +---------+-------+---------+
        | "Alice" |    34 | "Dave"  |
        | "Alice" |    34 | "Ernie" |
        | "Alice" |    34 | "Frank" |
        | "Bob"   |    51 | "Gaby"  |
        | "Bob"   |    51 | "Helen" |
        +---------+-------+---------+

        Notes
        -----
        Empty arrays or sets produce no rows in the resulting table. In the
        example above, notice that the name "Caroline" is not found in the
        exploded table.

        Missing arrays or sets are treated as empty.

        Parameters
        ----------
        field : :obj:`str` or :class:`.Expression`
            Top-level field name or expression.
        name : :obj:`str` or None
            If not `None`, rename the exploded field to `name`.

        Returns
        -------
        :class:`.Table`
            Table with exploded field.
        """

        if not isinstance(field, Expression):
            # field is a str
            field = self[field]

        if not field in self._fields_inverse:
            # nested or complex expression
            raise ValueError("method 'explode' expects a top-level field name or expression")
        if not field._indices == self._row_indices:
            # global field
            assert field._indices == self._global_indices
            raise ValueError("method 'explode' expects a field indexed by ['row'], found global field")

        if not isinstance(field.dtype, (tarray, tset)):
            raise ValueError(f"method 'explode' expects array or set, found: {field.dtype}")

        for k in self.key.values():
            if k is field:
                raise ValueError(f"method 'explode' cannot explode a key field")

        f = self._fields_inverse[field]
        t = Table(TableExplode(self._tir, f))
        if name is not None:
            t = t.rename({f: name})
        return t

    @typecheck_method(row_key=sequenceof(str),
                      col_key=sequenceof(str),
                      row_fields=sequenceof(str),
                      col_fields=sequenceof(str),
                      n_partitions=nullable(int))
    def to_matrix_table(self, row_key, col_key, row_fields=[], col_fields=[], n_partitions=None):
        """Construct a matrix table from a table in coordinate representation.

        Notes
        -----
        Any row fields in the table that do not appear in one of the arguments
        to this method are assumed to be entry fields of the resulting matrix
        table.

        Parameters
        ----------
        row_key : Sequence[str]
            Fields to be used as row key.
        col_key : Sequence[str]
            Fields to be used as column key.
        row_fields : Sequence[str]
            Fields to be stored once per row.
        col_fields : Sequence[str]
            Fields to be stored once per column.
        n_partitions : int or None
            Number of partitions.

        Returns
        -------
        :class:`.MatrixTable`
        """
        all_fields = list(itertools.chain(row_key, col_key, row_fields, col_fields))
        c = Counter(all_fields)
        row_field_set = set(self.row)
        for k, v in c.items():
            if k not in row_field_set:
                raise ValueError(f"'to_matrix_table': field {repr(k)} is not a row field")
            if v > 1:
                raise ValueError(f"'to_matrix_table': field {repr(k)} appeared in {v} field groups")

        if len(row_key) == 0:
            raise ValueError(f"'to_matrix_table': require at least one row key field")
        if len(col_key) == 0:
            raise ValueError(f"'to_matrix_table': require at least one col key field")

        return hl.MatrixTable._from_java(self._jt.jToMatrixTable(row_key,
                                                                 col_key,
                                                                 row_fields,
                                                                 col_fields,
                                                                 n_partitions))

    @property
    def globals(self) -> 'StructExpression':
        """Returns a struct expression including all global fields.

        Examples
        --------
        The data type of the globals struct:

        >>> table1.globals.dtype
        dtype('struct{global_field_1: int32, global_field_2: int32}')

        The number of global fields:

        >>> len(table1.globals)
        2

        Returns
        -------
        :class:`.StructExpression`
            Struct of all global fields.
        """
        return self._globals

    @property
    def row(self) -> 'StructExpression':
        """Returns a struct expression of all row-indexed fields, including keys.

        Examples
        --------
        The data type of the row struct:

        >>> table1.row.dtype
        dtype('struct{ID: int32, HT: int32, SEX: str, X: int32, Z: int32, C1: int32, C2: int32, C3: int32}')

        The number of row fields:

        >>> len(table1.row)
        8

        Returns
        -------
        :class:`.StructExpression`
            Struct of all row fields, including key fields.
        """
        return self._row

    @property
    def row_value(self) -> 'StructExpression':
        """Returns a struct expression including all non-key row-indexed fields.

        Examples
        --------
        The data type of the row struct:

        >>> table1.row_value.dtype
        dtype('struct{HT: int32, SEX: str, X: int32, Z: int32, C1: int32, C2: int32, C3: int32}')

        The number of row fields:

        >>> len(table1.row_value)
        7

        Returns
        -------
        :class:`.StructExpression`
            Struct of all non-key row fields.
        """
        return self._row.drop(*self.key.keys())

    @staticmethod
    @typecheck(df=pyspark.sql.DataFrame,
               key=table_key_type)
    def from_spark(df, key=[]):
        """Convert PySpark SQL DataFrame to a table.

        Examples
        --------

        >>> t = Table.from_spark(df) # doctest: +SKIP

        Notes
        -----

        Spark SQL data types are converted to Hail types as follows:

        .. code-block:: text

          BooleanType => :py:data:`.tbool`
          IntegerType => :py:data:`.tint32`
          LongType => :py:data:`.tint64`
          FloatType => :py:data:`.tfloat32`
          DoubleType => :py:data:`.tfloat64`
          StringType => :py:data:`.tstr`
          BinaryType => :class:`.TBinary`
          ArrayType => :class:`.tarray`
          StructType => :class:`.tstruct`

        Unlisted Spark SQL data types are currently unsupported.

        Parameters
        ----------
        df : :class:`.pyspark.sql.DataFrame`
            PySpark DataFrame.
        
        key : :obj:`str` or :obj:`list` of :obj:`str`
            Key fields.

        Returns
        -------
        :class:`.Table`
            Table constructed from the Spark SQL DataFrame.
        """

        return Table._from_java(Env.hail().table.Table.fromDF(Env.hc()._jhc, df._jdf, key))

    @typecheck_method(flatten=bool)
    def to_spark(self, flatten=True):
        """Converts this table to a Spark DataFrame.

        Because Spark cannot represent complex types, types are
        expanded before flattening or conversion.

        Parameters
        ----------
        flatten : :obj:`bool`
            If ``True``, :meth:`flatten` before converting to Spark DataFrame.

        Returns
        -------
        :class:`.pyspark.sql.DataFrame`

        """
        t = self.expand_types()
        if flatten:
            t = t.flatten()
        return pyspark.sql.DataFrame(t._jt.toDF(Env.hc()._jsql_context), Env.sql_context())

    @typecheck_method(flatten=bool)
    def to_pandas(self, flatten=True):
        """Converts this table to a Pandas DataFrame.

        Because conversion to Pandas is done through Spark, and Spark
        cannot represent complex types, types are expanded before
        flattening or conversion.

        Parameters
        ----------
        flatten : :obj:`bool`
            If ``True``, :meth:`flatten` before converting to Pandas DataFrame.

        Returns
        -------
        :class:`.pandas.DataFrame`

        """
        return self.to_spark(flatten).toPandas()

    @staticmethod
    @typecheck(df=pandas.DataFrame,
               key=oneof(str, sequenceof(str)))
    def from_pandas(df, key=[]):
        """Create table from Pandas DataFrame

        Examples
        --------

        >>> t = hl.Table.from_pandas(df) # doctest: +SKIP

        Parameters
        ----------
        df : :class:`.pandas.DataFrame`
            Pandas DataFrame.
        key : :obj:`str` or :obj:`list` of :obj:`str`
            Key fields.

        Returns
        -------
        :class:`.Table`
        """
        return Table.from_spark(Env.sql_context().createDataFrame(df), key)

    @typecheck_method(other=table_type, tolerance=nullable(numeric), absolute=bool)
    def _same(self, other, tolerance=1e-6, absolute=False):
        return self._jt.same(other._jt, tolerance, absolute)

    def collect_by_key(self, name: str= 'values') -> 'Table':
        """Collect values for each unique key into an array.

        .. include:: _templates/req_keyed_table.rst

        Examples
        --------
        >>> t1 = hl.Table.parallelize([
        ...     {'t': 'foo', 'x': 4, 'y': 'A'},
        ...     {'t': 'bar', 'x': 2, 'y': 'B'},
        ...     {'t': 'bar', 'x': -3, 'y': 'C'},
        ...     {'t': 'quam', 'x': 0, 'y': 'D'}],
        ...     hl.tstruct(t=hl.tstr, x=hl.tint32, y=hl.tstr),
        ...     key='t')

        >>> t1.show()
        +--------+-------+-----+
        | t      |     x | y   |
        +--------+-------+-----+
        | str    | int32 | str |
        +--------+-------+-----+
        | "bar"  |     2 | "B" |
        | "bar"  |    -3 | "C" |
        | "foo"  |     4 | "A" |
        | "quam" |     0 | "D" |
        +--------+-------+-----+

        >>> t1.collect_by_key().show()
        +--------+---------------------------------+
        | t      | values                          |
        +--------+---------------------------------+
        | str    | array<struct{x: int32, y: str}> |
        +--------+---------------------------------+
        | "bar"  | [(2,"B"),(-3,"C")]              |
        | "foo"  | [(4,"A")]                       |
        | "quam" | [(0,"D")]                       |
        +--------+---------------------------------+

        Notes
        -----
        The order of the values array is not guaranteed.

        Parameters
        ----------
        name : :obj:`str`
            Field name for all values per key.

        Returns
        -------
        :class:`.Table`
        """

        return Table._from_java(self._jt.groupByKey(name))

    def distinct(self) -> 'Table':
        """Keep only one row for each unique key.

        .. include:: _templates/req_keyed_table.rst

        Examples
        --------
        >>> t1 = hl.Table.parallelize([
        ...     {'a': 'foo', 'b': 1},
        ...     {'a': 'bar', 'b': 5},
        ...     {'a': 'bar', 'b': 2}],
        ...     hl.tstruct(a=hl.tstr, b=hl.tint32),
        ...     key='a')

        >>> t1.show()
        +-------+-------+
        | a     |     b |
        +-------+-------+
        | str   | int32 |
        +-------+-------+
        | "bar" |     5 |
        | "bar" |     2 |
        | "foo" |     1 |
        +-------+-------+

        >>> t1.distinct().show()
        +-------+-------+
        | a     |     b |
        +-------+-------+
        | str   | int32 |
        +-------+-------+
        | "bar" |     5 |
        | "foo" |     1 |
        +-------+-------+

        Notes
        -----
        The row chosen per distinct key is not guaranteed.

        Returns
        -------
        :class:`.Table`
        """

        return Table(TableDistinct(self._tir))

    @typecheck_method(parts=sequenceof(int), keep=bool)
    def _filter_partitions(self, parts, keep=True):
        return Table._from_java(self._jt.filterPartitions(parts, keep))

    @typecheck_method(cols=table_type, entries_field_name=str)
    def _unlocalize_entries(self, cols, entries_field_name):
        return hl.MatrixTable._from_java(self._jt.unlocalizeEntries(cols._jt, entries_field_name))

    @staticmethod
    @typecheck(tables=sequenceof(table_type), data_field_name=str, global_field_name=str)
    def _multi_way_zip_join(tables, data_field_name, global_field_name):
        if not tables:
            raise ValueError('multi_way_zip_join must have at least one table as an argument')
        head = tables[0]
        if any(head.key.dtype != t.key.dtype for t in tables):
            raise TypeError('All input tables to multi_way_zip_join must have the same key type')
        if any(head.row.dtype != t.row.dtype for t in tables):
            raise TypeError('All input tables to multi_way_zip_join must have the same row type')
        if any(head.globals.dtype != t.globals.dtype for t in tables):
            raise TypeError('All input tables to multi_way_zip_join must have the same global type')
        jt = Env.hail().table.Table.multiWayZipJoin([t._jt for t in tables],
                                                    data_field_name,
                                                    global_field_name)
        return Table._from_java(jt)

table_type.set(Table)
