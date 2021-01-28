import collections
import itertools
import pandas
import pyspark
from typing import Optional, Dict, Callable

from hail.expr.expressions import Expression, StructExpression, \
    BooleanExpression, expr_struct, expr_any, expr_bool, analyze, Indices, \
    construct_reference, to_expr, construct_expr, extract_refs_by_indices, \
    ExpressionException, TupleExpression, unify_all, NumericExpression, \
    StringExpression, CallExpression, CollectionExpression, DictExpression, \
    IntervalExpression, LocusExpression, NDArrayExpression, expr_array
from hail.expr.types import hail_type, tstruct, types_match, tarray, tset
from hail.expr.table_type import ttable
import hail.ir as ir
from hail.typecheck import typecheck, typecheck_method, dictof, anytype, \
    anyfunc, nullable, sequenceof, oneof, numeric, lazy, enumeration, \
    table_key_type, func_spec
from hail.utils import deduplicate
from hail.utils.placement_tree import PlacementTree
from hail.utils.java import Env, info, warning
from hail.utils.misc import wrap_to_tuple, storage_level, plural, \
    get_nice_field_error, get_nice_attr_error, get_key_by_exprs, check_keys, \
    get_select_exprs, check_annotate_exprs, process_joins
import hail as hl

table_type = lazy()


class TableIndexKeyError(Exception):
    def __init__(self, key_type, index_expressions):
        super().__init__()
        self.key_type = key_type
        self.index_expressions = index_expressions


class Ascending:
    def __init__(self, col):
        self.col = col

    def __eq__(self, other):
        return isinstance(other, Ascending) and self.col == other.col

    def __ne__(self, other):
        return not self == other


class Descending:
    def __init__(self, col):
        self.col = col

    def __eq__(self, other):
        return isinstance(other, Descending) and self.col == other.col

    def __ne__(self, other):
        return not self == other


@typecheck(col=oneof(Expression, str))
def asc(col):
    """Sort by `col` ascending."""

    return Ascending(col)


@typecheck(col=oneof(Expression, str))
def desc(col):
    """Sort by `col` descending."""

    return Descending(col)


class ExprContainer:

    # this can only grow as big as the object dir, so no need to worry about memory leak
    _warned_about = set()

    def __init__(self):
        self._fields: Dict[str, Expression] = {}
        self._fields_inverse: Dict[Expression, str] = {}
        self._dir = set(dir(self))
        super(ExprContainer, self).__init__()

    def _set_field(self, key, value):
        assert key not in self._fields_inverse, key
        self._fields[key] = value
        self._fields_inverse[value] = key

        # key is in __dir for methods
        # key is in __dict__ for private class fields
        if key in self._dir or key in self.__dict__:
            if key not in ExprContainer._warned_about:
                ExprContainer._warned_about.add(key)
                warning(f"Name collision: field {repr(key)} already in object dict. "
                        f"\n  This field must be referenced with __getitem__ syntax: obj[{repr(key)}]")
        else:
            self.__dict__[key] = value

    def _get_field(self, item) -> Expression:
        if item in self._fields:
            return self._fields[item]

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

        raise AttributeError(get_nice_attr_error(self, item))

    def _copy_fields_from(self, other: 'ExprContainer'):
        self._fields = other._fields
        self._fields_inverse = other._fields_inverse


class GroupedTable(ExprContainer):
    """Table grouped by row that can be aggregated into a new table.

    There are only two operations on a grouped table, :meth:`.GroupedTable.partition_hint`
    and :meth:`.GroupedTable.aggregate`.
    """

    def __init__(self, parent: 'Table', key_expr):
        super(GroupedTable, self).__init__()
        self._key_expr = key_expr
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
        ...                       .aggregate(meanX = hl.agg.mean(table1.X), sumZ = hl.agg.sum(table1.Z)))

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
    def aggregate(self, **named_exprs) -> 'Table':
        """Aggregate by group, used after :meth:`.Table.group_by`.

        Examples
        --------
        Compute the mean value of `X` and the sum of `Z` per unique `ID`:

        >>> table_result = (table1.group_by(table1.ID)
        ...                       .aggregate(meanX = hl.agg.mean(table1.X), sumZ = hl.agg.sum(table1.Z)))

        Group by a height bin and compute sex ratio per bin:

        >>> table_result = (table1.group_by(height_bin = table1.HT // 20)
        ...                       .aggregate(fraction_female = hl.agg.fraction(table1.SEX == 'F')))

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
        for name, expr in named_exprs.items():
            analyze(f'GroupedTable.aggregate: ({repr(name)})', expr, self._parent._global_indices, {self._parent._row_axis})
        if not named_exprs.keys().isdisjoint(set(self._key_expr)):
            intersection = set(named_exprs.keys()) & set(self._key_expr)
            raise ValueError(
                f'GroupedTable.aggregate: Group names and aggregration expression names overlap: {intersection}')

        base, _ = self._parent._process_joins(self._key_expr, *named_exprs.values())

        key_struct = self._key_expr
        return Table(ir.TableKeyByAndAggregate(base._tir,
                                               hl.struct(**named_exprs)._ir,
                                               key_struct._ir,
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
    ...    return hl.if_else(sex == 'M',
    ...                     (height - height_mean_m) / height_sd_m,
    ...                     (height - height_mean_f) / height_sd_f)
    >>>
    >>> table1 = table1.annotate(height_z = get_z(table1.HT, table1.SEX))
    >>> table1 = table1.annotate_globals(global_field_1 = [1, 2, 3])

    Filter rows of the table:

    >>> table2 = table2.filter(table2.B != 'rabbit')

    Compute global aggregation statistics:

    >>> t1_stats = table1.aggregate(hl.struct(mean_c1 = hl.agg.mean(table1.C1),
    ...                                       mean_c2 = hl.agg.mean(table1.C2),
    ...                                       stats_c3 = hl.agg.stats(table1.C3)))
    >>> print(t1_stats)

    Group by a field and aggregate to produce a new table:

    >>> table3 = (table1.group_by(table1.SEX)
    ...                 .aggregate(mean_height_data = hl.agg.mean(table1.HT)))
    >>> table3.show()

    Join tables together inside an annotation expression:

    >>> table2 = table2.key_by('ID')
    >>> table1 = table1.annotate(B = table2[table1.ID].B)
    >>> table1.show()
    """

    @staticmethod
    def _from_java(jtir):
        return Table(ir.JavaTable(jtir))

    def __init__(self, tir):
        super(Table, self).__init__()

        self._tir = tir
        self._type = self._tir.typ

        self._row_axis = 'row'

        self._global_indices = Indices(axes=set(), source=self)
        self._row_indices = Indices(axes={self._row_axis}, source=self)

        self._global_type = self._type.global_type
        self._row_type = self._type.row_type

        self._globals = construct_reference('global', self._global_type, indices=self._global_indices)
        self._row = construct_reference('row', self._row_type, indices=self._row_indices)

        self._indices_from_ref = {'global': self._global_indices,
                                  'row': self._row_indices}

        self._key = hl.struct(
            **{k: self._row[k] for k in self._type.row_key})

        for k, v in itertools.chain(self._globals.items(),
                                    self._row.items()):
            self._set_field(k, v)

    @property
    def _schema(self) -> ttable:
        return ttable(self._global_type, self._row_type, list(self._key))

    def __getitem__(self, item):
        if isinstance(item, str):
            return self._get_field(item)

        try:
            return self.index(*wrap_to_tuple(item))
        except TypeError as e:
            raise TypeError("Table.__getitem__: invalid index argument(s)\n"
                            "  Usage 1: field selection: ht['field']\n"
                            "  Usage 2: Left distinct join: ht[ht2.key] or ht[ht2.field1, ht2.field2]") from e

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

    @property
    def _value(self) -> 'StructExpression':
        return self.row.drop(*self.key)

    def n_partitions(self):
        """Returns the number of partitions in the table.

        Returns
        -------
        :obj:`int`
        """
        return Env.backend().execute(ir.TableToValueApply(self._tir, {'name': 'NPartitionsTable'}))

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
        return Env.backend().execute(ir.TableCount(self._tir))

    def _force_count(self):
        return Env.backend().execute(ir.TableToValueApply(self._tir, {'name': 'ForceCountTable'}))

    @typecheck_method(caller=str,
                      row=expr_struct())
    def _select(self, caller, row) -> 'Table':
        analyze(caller, row, self._row_indices)
        base, cleanup = self._process_joins(row)
        return cleanup(Table(ir.TableMapRows(base._tir, row._ir)))

    @typecheck_method(caller=str, s=expr_struct())
    def _select_globals(self, caller, s) -> 'Table':
        base, cleanup = self._process_joins(s)
        analyze(caller, s, self._global_indices)
        return cleanup(Table(ir.TableMapGlobals(base._tir, s._ir)))

    @classmethod
    @typecheck_method(rows=anytype,
                      schema=nullable(hail_type),
                      key=table_key_type,
                      n_partitions=nullable(int))
    def parallelize(cls, rows, schema=None, key=None, n_partitions=None) -> 'Table':
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
        schema : str or a hail type (see :ref:`hail_types`), optional
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
        table = Table(ir.TableParallelize(ir.MakeStruct([
            ('rows', rows._ir),
            ('global', ir.MakeStruct([]))]), n_partitions))
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
        keys : varargs of type :class:`str`
            Field(s) to key by.

        Returns
        -------
        :class:`.Table`
            Table with a new key.
        """
        key_fields, computed_keys = get_key_by_exprs("Table.key_by", keys, named_keys, self._row_indices)

        if not computed_keys:
            return Table(ir.TableKeyBy(self._tir, key_fields))

        new_row = self.row.annotate(**computed_keys)
        base, cleanup = self._process_joins(new_row)

        return cleanup(Table(
            ir.TableKeyBy(
                ir.TableMapRows(
                    ir.TableKeyBy(base._tir, []),
                    new_row._ir),
                list(key_fields))))

    @typecheck_method(keys=oneof(str, expr_any),
                      named_keys=expr_any)
    def _key_by_assert_sorted(self, *keys, **named_keys) -> 'Table':
        key_fields, computed_keys = get_key_by_exprs("Table.key_by", keys, named_keys, self._row_indices)

        if not computed_keys:
            return Table(ir.TableKeyBy(self._tir, key_fields, is_sorted=True))
        else:
            new_row = self.row.annotate(**computed_keys)
            base, cleanup = self._process_joins(new_row)

            return cleanup(Table(
                ir.TableKeyBy(
                    ir.TableMapRows(
                        ir.TableKeyBy(base._tir, []),
                        new_row._ir),
                    list(key_fields),
                    is_sorted=True)))

    @typecheck_method(named_exprs=expr_any)
    def annotate_globals(self, **named_exprs) -> 'Table':
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
        caller = 'Table.annotate_globals'
        check_annotate_exprs(caller, named_exprs, self._global_indices, set())
        return self._select_globals('Table.annotate_globals', self.globals.annotate(**named_exprs))

    def select_globals(self, *exprs, **named_exprs) -> 'Table':
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
        exprs : variable-length args of :class:`str` or :class:`.Expression`
            Arguments that specify field names or nested field reference expressions.
        named_exprs : keyword args of :class:`.Expression`
            Field names and the expressions to compute them.

        Returns
        -------
        :class:`.Table`
            Table with specified global fields.
        """
        caller = 'Table.select_globals'
        new_globals = get_select_exprs(caller,
                                       exprs,
                                       named_exprs,
                                       self._global_indices,
                                       self._globals)

        return self._select_globals(caller, new_globals)

    @typecheck_method(named_exprs=expr_any)
    def transmute_globals(self, **named_exprs) -> 'Table':
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
        check_annotate_exprs(caller, named_exprs, self._global_indices, set())
        fields_referenced = extract_refs_by_indices(named_exprs.values(), self._global_indices) - set(named_exprs.keys())

        return self._select_globals(caller,
                                    self.globals.annotate(**named_exprs).drop(*fields_referenced))

    @typecheck_method(named_exprs=expr_any)
    def transmute(self, **named_exprs) -> 'Table':
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
        check_annotate_exprs(caller, named_exprs, self._row_indices, set())
        fields_referenced = extract_refs_by_indices(named_exprs.values(), self._row_indices) - set(named_exprs.keys())
        fields_referenced -= set(self.key)

        return self._select(caller, self.row.annotate(**named_exprs).drop(*fields_referenced))

    @typecheck_method(named_exprs=expr_any)
    def annotate(self, **named_exprs) -> 'Table':
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
        check_annotate_exprs(caller, named_exprs, self._row_indices, set())
        return self._select(caller, self.row.annotate(**named_exprs))

    @typecheck_method(expr=expr_bool,
                      keep=bool)
    def filter(self, expr, keep=True) -> 'Table':
        """Filter rows.

        Examples
        --------

        Keep rows where ``C1`` equals 5:

        >>> table_result = table1.filter(table1.C1 == 5)

        Remove rows where ``C1`` equals 10:

        >>> table_result = table1.filter(table1.C1 == 10, keep=False)

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

        return cleanup(Table(ir.TableFilter(base._tir, ir.filter_predicate_with_keep(expr._ir, keep))))

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
        exprs : variable-length args of :class:`str` or :class:`.Expression`
            Arguments that specify field names or nested field reference expressions.
        named_exprs : keyword args of :class:`.Expression`
            Field names and the expressions to compute them.

        Returns
        -------
        :class:`.Table`
            Table with specified fields.
        """
        row = get_select_exprs('Table.select',
                               exprs,
                               named_exprs,
                               self._row_indices,
                               self._row)

        return self._select('Table.select', row)

    @typecheck_method(exprs=oneof(str, Expression))
    def drop(self, *exprs) -> 'Table':
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
        exprs : varargs of :class:`str` or :class:`.Expression`
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
            table = table._select_globals('drop',
                                          self._globals.drop(*[f for f in table.globals if f in fields_to_drop]))

        if any(self._fields[field]._indices == self._row_indices for field in fields_to_drop):
            # need to drop row fields
            protected_key = set(self._row_indices.protected_key)
            for f in fields_to_drop:
                check_keys('drop', f, protected_key)
            row_fields = set(table.row)
            to_drop = [f for f in fields_to_drop if f in row_fields]
            table = table._select('drop', table.row.drop(*to_drop))

        return table

    @typecheck_method(output=str,
                      types_file=nullable(str),
                      header=bool,
                      parallel=nullable(ir.ExportType.checker),
                      delimiter=str)
    def export(self, output, types_file=None, header=True, parallel=None, delimiter='\t'):
        """Export to a text file.

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

        Nested structures will be exported as JSON. In order to export nested struct
        fields as separate fields in the resulting table, use :meth:`flatten` first.

        Warning
        -------
        Do not export to a path that is being read from in the same pipeline.

        See Also
        --------
        :meth:`flatten`, :meth:`write`

        Parameters
        ----------
        output : :class:`str`
            URI at which to write exported file.
        types_file : :class:`str`, optional
            URI at which to write file containing field type information.
        header : :obj:`bool`
            Include a header in the file.
        parallel : :class:`str`, optional
            If None, a single file is produced, otherwise a
            folder of file shards is produced. If 'separate_header',
            the header file is output separately from the file shards. If
            'header_per_shard', each file shard has a header. If set to None
            the export will be slower.
        delimiter : :class:`str`
            Field delimiter.
        """
        parallel = ir.ExportType.default(parallel)
        Env.backend().execute(
            ir.TableWrite(self._tir, ir.TableTextWriter(output, types_file, header, parallel, delimiter)))

    def group_by(self, *exprs, **named_exprs) -> 'GroupedTable':
        """Group by a new key for use with :meth:`.GroupedTable.aggregate`.

        Examples
        --------
        Compute the mean value of `X` and the sum of `Z` per unique `ID`:

        >>> table_result = (table1.group_by(table1.ID)
        ...                       .aggregate(meanX = hl.agg.mean(table1.X), sumZ = hl.agg.sum(table1.Z)))

        Group by a height bin and compute sex ratio per bin:

        >>> table_result = (table1.group_by(height_bin = table1.HT // 20)
        ...                       .aggregate(fraction_female = hl.agg.fraction(table1.SEX == 'F')))

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
        ...                       .aggregate(meanX = hl.agg.mean(table1.X)))

        Second, field reference variable-length arguments:

        >>> table_result = (table1.group_by(table1.C1, table1.C2)
        ...                       .aggregate(meanX = hl.agg.mean(table1.X)))

        Last, expression keyword arguments:

        >>> table_result = (table1.group_by(C1 = table1.C1, C2 = table1.C2)
        ...                       .aggregate(meanX = hl.agg.mean(table1.X)))

        Additionally, the variable-length argument syntax also permits nested field
        references. Given the following struct field `s`:

        >>> table3 = table1.annotate(s = hl.struct(x=table1.X, z=table1.Z))

        The following two usages are equivalent, grouping by one field, `x`:

        >>> table_result = (table3.group_by(table3.s.x)
        ...                       .aggregate(meanX = hl.agg.mean(table3.X)))

        >>> table_result = (table3.group_by(x = table3.s.x)
        ...                       .aggregate(meanX = hl.agg.mean(table3.X)))

        The keyword argument syntax permits arbitrary expressions:

        >>> table_result = (table1.group_by(foo=table1.X ** 2 + 1)
        ...                       .aggregate(meanZ = hl.agg.mean(table1.Z)))

        These syntaxes can be mixed together, with the stipulation that all keyword arguments
        must come at the end due to Python language restrictions.

        >>> table_result = (table1.group_by(table1.C1, 'C2', height_bin = table1.HT // 20)
        ...                       .aggregate(meanX = hl.agg.mean(table1.X)))

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
        key, computed_key = get_key_by_exprs('Table.group_by',
                                             exprs,
                                             named_exprs,
                                             self._row_indices,
                                             override_protected_indices={self._global_indices})
        return GroupedTable(self, self.row.annotate(**computed_key).select(*key))

    @typecheck_method(expr=expr_any, _localize=bool)
    def aggregate(self, expr, _localize=True):
        """Aggregate over rows into a local value.

        Examples
        --------
        Aggregate over rows:

        >>> table1.aggregate(hl.struct(fraction_male=hl.agg.fraction(table1.SEX == 'M'),
        ...                            mean_x=hl.agg.mean(table1.X)))
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

        agg_ir = ir.TableAggregate(base._tir, expr._ir)

        if _localize:
            return Env.backend().execute(agg_ir)

        return construct_expr(ir.LiftMeOut(agg_ir), expr.dtype)

    @typecheck_method(output=str,
                      overwrite=bool,
                      stage_locally=bool,
                      _codec_spec=nullable(str),
                      _read_if_exists=bool,
                      _intervals=nullable(sequenceof(anytype)),
                      _filter_intervals=bool)
    def checkpoint(self, output: str, overwrite: bool = False, stage_locally: bool = False,
                   _codec_spec: Optional[str] = None, _read_if_exists: bool = False,
                   _intervals=None, _filter_intervals=False) -> 'Table':
        """Checkpoint the table to disk by writing and reading.

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
        :class:`Table`


        .. include:: _templates/write_warning.rst

        Notes
        -----
        An alias for :meth:`write` followed by :func:`.read_table`. It is
        possible to read the file at this path later with :func:`.read_table`.

        Examples
        --------
        >>> table1 = table1.checkpoint('output/table_checkpoint.ht')

        """
        if _codec_spec is None:
            _codec_spec = """{
  "name": "LEB128BufferSpec",
  "child": {
    "name": "BlockingBufferSpec",
    "blockSize": 32768,
    "child": {
      "name": "LZ4FastBlockBufferSpec",
      "blockSize": 32768,
      "child": {
        "name": "StreamBlockBufferSpec"
      }
    }
  }
}"""

        if not _read_if_exists or not hl.hadoop_exists(f'{output}/_SUCCESS'):
            self.write(output=output, overwrite=overwrite, stage_locally=stage_locally, _codec_spec=_codec_spec)
        return hl.read_table(output, _intervals=_intervals, _filter_intervals=_filter_intervals)

    @typecheck_method(output=str,
                      overwrite=bool,
                      stage_locally=bool,
                      _codec_spec=nullable(str))
    def write(self, output: str, overwrite=False, stage_locally: bool = False,
              _codec_spec: Optional[str] = None):
        """Write to disk.

        Examples
        --------

        >>> table1.write('output/table1.ht')

        .. include:: _templates/write_warning.rst

        See Also
        --------
        :func:`.read_table`

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

        Env.backend().execute(ir.TableWrite(self._tir, ir.TableNativeWriter(output, overwrite, stage_locally, _codec_spec)))

    def _show(self, n, width, truncate, types):
        return Table._Show(self, n, width, truncate, types)

    class _Show:
        def __init__(self, table, n, width, truncate, types):
            if n is None or width is None:
                import shutil
                (columns, lines) = shutil.get_terminal_size((80, 10))
                width = width or columns
                n = n or min(max(10, (lines - 20)), 100)
            self.table = table
            self.n = n
            self.width = max(width, 8)
            if truncate:
                self.truncate = min(max(truncate, 4), width - 4)
            else:
                self.truncate = width - 4
            self.types = types
            self._data = None

        def __str__(self):
            return self._ascii_str()

        def __repr__(self):
            return self.__str__()

        def data(self):
            if self._data is None:
                t = self.table.flatten()
                row_dtype = t.row.dtype
                t = t.select(**{k: hl._showstr(v) for (k, v) in t.row.items()})
                rows, has_more = t._take_n(self.n)
                self._data = (rows, has_more, row_dtype)
            return self._data

        def _repr_html_(self):
            return self._html_str()

        def _ascii_str(self):
            truncate = self.truncate
            types = self.types

            def trunc(s):
                if len(s) > truncate:
                    return s[:truncate - 3] + "..."
                return s

            rows, has_more, dtype = self.data()
            fields = list(dtype)
            trunc_fields = [trunc(f) for f in fields]
            n_fields = len(fields)

            type_strs = [trunc(str(dtype[f])) for f in fields] if types else [''] * len(fields)
            right_align = [hl.expr.types.is_numeric(dtype[f]) for f in fields]

            rows = [[trunc(row[f]) for f in fields] for row in rows]

            def max_value_width(i):
                return max(itertools.chain([0], (len(row[i]) for row in rows)))

            column_width = [max(len(trunc_fields[i]), len(type_strs[i]), max_value_width(i)) for i in range(n_fields)]

            column_blocks = []
            start = 0
            i = 1
            w = column_width[0] + 4 if column_width else 0
            while i < n_fields:
                w = w + column_width[i] + 3
                if w > self.width:
                    column_blocks.append((start, i))
                    start = i
                    w = column_width[i] + 4
                i = i + 1
            column_blocks.append((start, i))

            def format_hline(widths):
                if not widths:
                    return "++\n"
                return '+-' + '-+-'.join(['-' * w for w in widths]) + '-+\n'

            def pad(v, w, ra):
                e = w - len(v)
                if ra:
                    return ' ' * e + v
                else:
                    return v + ' ' * e

            def format_line(values, widths, right_align):
                if not values:
                    return "||\n"
                values = map(pad, values, widths, right_align)
                return '| ' + ' | '.join(values) + ' |\n'

            s = ''
            first = True
            for (start, end) in column_blocks:
                if first:
                    first = False
                else:
                    s += '\n'

                block_column_width = column_width[start:end]
                block_right_align = right_align[start:end]
                hline = format_hline(block_column_width)

                s += hline
                s += format_line(trunc_fields[start:end], block_column_width, block_right_align)
                s += hline
                if types:
                    s += format_line(type_strs[start:end], block_column_width, block_right_align)
                    s += hline
                for row in rows:
                    row = row[start:end]
                    s += format_line(row, block_column_width, block_right_align)
                s += hline

            if has_more:
                n_rows = len(rows)
                s += f"showing top { n_rows } { 'row' if n_rows == 1 else 'rows' }\n"

            return s

        def _html_str(self):
            import html
            types = self.types

            rows, has_more, dtype = self.data()
            fields = list(dtype)

            default_td_style = ('white-space: nowrap; '
                                'max-width: 500px; '
                                'overflow: hidden; '
                                'text-overflow: ellipsis; ')

            def format_line(values, extra_style=''):
                style = default_td_style + extra_style
                return (f'<tr><td style="{style}">' + f'</td><td style="{style}">'.join(values) + '</td></tr>\n')

            arranged_field_names = PlacementTree.from_named_type('row', self.table.row.dtype)

            s = '<table>'
            s += '<thead>'
            for header_row in arranged_field_names.to_grid():
                s += '<tr>'
                div_style = 'text-align: left;'
                non_empty_div_style = 'border-bottom: solid 2px #000; padding-bottom: 5px'
                for header_cell in header_row:
                    text, width = header_cell
                    s += f'<td style="{default_td_style}" colspan="{width}">'
                    if text is not None:
                        s += f'<div style="{div_style}{non_empty_div_style}">'
                        s += text
                        s += '</div>'
                    else:
                        s += f'<div style="{div_style}"></div>'
                    s += '</td>'
                s += '</tr>'
            if types:
                s += format_line([html.escape(str(dtype[f])) for f in fields],
                                 extra_style="text-align: left;")
            s += '</thead><tbody>'
            for row in rows:
                s += format_line([html.escape(row[f]) for f in row])
            s += '</tbody></table>'

            if has_more:
                n_rows = len(rows)
                s += '<p style="background: #fdd; padding: 0.4em;">'
                s += f"showing top { n_rows } { plural('row', n_rows) }"
                s += '</p>\n'

            return s

    def _take_n(self, n):
        if n < 0:
            rows = self.collect()
            has_more = False
        else:
            rows = self.take(n + 1)
            has_more = len(rows) > n
            rows = rows[:n]
        return rows, has_more

    @staticmethod
    def _hl_format(v, truncate):
        return hl._showstr(v, truncate)

    @typecheck_method(n=nullable(int), width=nullable(int), truncate=nullable(int), types=bool, handler=nullable(anyfunc), n_rows=nullable(int))
    def show(self, n=None, width=None, truncate=None, types=True, handler=None, n_rows=None):
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

        Notes
        -----
        The output can be passed piped to another output source using the `handler` argument:

        >>> ht.show(handler=lambda x: logging.info(x))  # doctest: +SKIP

        Parameters
        ----------
        n or n_rows : :obj:`int`
            Maximum number of rows to show, or negative to show all rows.
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
        if n_rows is not None and n is not None:
            raise ValueError(f'specify one of n_rows or n, received {n_rows} and {n}')
        if n_rows is not None:
            n = n_rows
        del n_rows
        if handler is None:
            handler = hl.utils.default_handler()
        handler(self._show(n, width, truncate, types))

    def index(self, *exprs, all_matches=False) -> 'Expression':
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
        all_matches : bool
            Experimental. If ``True``, value of expression is array of all matches.

        Returns
        -------
        :class:`.Expression`
        """
        try:
            return self._index(*exprs, all_matches=all_matches)
        except TableIndexKeyError as err:
            raise ExpressionException(f"Key type mismatch: cannot index table with given expressions:\n"
                                      f"  Table key:         {', '.join(str(t) for t in err.key_type.values()) or '<<<empty key>>>'}\n"
                                      f"  Index Expressions: {', '.join(str(e.dtype) for e in err.index_expressions)}")

    @staticmethod
    def _maybe_truncate_for_flexindex(indexer, indexee_dtype):
        if not len(indexee_dtype) > 0:
            raise ValueError('Must have non-empty key to index')

        if not isinstance(indexer.dtype, (hl.tstruct, hl.ttuple)):
            indexer = hl.tuple([indexer])

        matching_prefix = 0
        for x, y in zip(indexer.dtype.types, indexee_dtype.types):
            if x != y:
                break
            matching_prefix += 1
        prefix_match = matching_prefix == len(indexee_dtype)
        direct_match = prefix_match and \
            len(indexer) == len(indexee_dtype)
        prefix_interval_match = len(indexee_dtype) == 1 and \
            isinstance(indexee_dtype[0], hl.tinterval) and \
            indexer.dtype[0] == indexee_dtype[0].point_type
        direct_interval_match = prefix_interval_match and \
            len(indexer) == 1
        if direct_match or direct_interval_match:
            return indexer
        if prefix_match:
            return indexer[0:matching_prefix]
        if prefix_interval_match:
            return indexer[0]
        return None

    @typecheck_method(indexer=expr_any, all_matches=bool)
    def _maybe_flexindex_table_by_expr(self, indexer, all_matches=False):
        truncated_indexer = Table._maybe_truncate_for_flexindex(
            indexer, self.key.dtype)
        if truncated_indexer is not None:
            return self.index(truncated_indexer, all_matches=all_matches)
        return None

    def _index(self, *exprs, all_matches=False) -> 'Expression':
        exprs = tuple(exprs)
        if not len(exprs) > 0:
            raise ValueError('Require at least one expression to index')
        non_exprs = list(filter(lambda e: not isinstance(e, Expression), exprs))
        if non_exprs:
            raise TypeError(f"Index arguments must be expressions, found {non_exprs}")

        from hail.matrixtable import MatrixTable
        indices, aggregations = unify_all(*exprs)
        src = indices.source

        if src is None or len(indices.axes) == 0:
            # FIXME: this should be OK: table[m.global_index_into_table]
            raise ExpressionException('Cannot index with a scalar expression')

        is_interval = (len(exprs) == 1
                       and len(self.key) > 0
                       and isinstance(self.key[0].dtype, hl.tinterval)
                       and exprs[0].dtype == self.key[0].dtype.point_type)

        if not types_match(list(self.key.values()), list(exprs)):
            if (len(exprs) == 1
                    and isinstance(exprs[0], TupleExpression)):
                return self._index(*exprs[0], all_matches=all_matches)

            if (len(exprs) == 1
                    and isinstance(exprs[0], StructExpression)):
                return self._index(*exprs[0].values(), all_matches=all_matches)

            if not is_interval:
                raise TableIndexKeyError(self.key.dtype, exprs)

        uid = Env.get_uid()

        if all_matches and not is_interval:
            return self.collect_by_key(uid).index(*exprs)[uid]

        new_schema = self.row_value.dtype
        if all_matches:
            new_schema = hl.tarray(new_schema)

        if isinstance(src, Table):
            for e in exprs:
                analyze('Table.index', e, src._row_indices)

            is_key = len(src.key) >= len(exprs) and all(expr is key_field for expr, key_field in zip(exprs, src.key.values()))

            if not is_key:
                uids = [Env.get_uid() for i in range(len(exprs))]
                all_uids = uids[:]
            else:
                all_uids = []

            def joiner(left):
                if not is_key:
                    original_key = list(left.key)
                    left = Table(ir.TableMapRows(left.key_by()._tir,
                                                 ir.InsertFields(left._row._ir,
                                                                 list(zip(uids, [e._ir for e in exprs])),
                                                                 None))).key_by(*uids)

                    def rekey_f(t):
                        return t.key_by(*original_key)
                else:
                    def rekey_f(t):
                        return t

                if is_interval:
                    left = Table(ir.TableIntervalJoin(left._tir, self._tir, uid, all_matches))
                else:
                    left = Table(ir.TableLeftJoinRightDistinct(left._tir, self._tir, uid))
                return rekey_f(left)

            all_uids.append(uid)
            join_ir = ir.Join(ir.GetField(ir.TopLevelReference('row'), uid),
                              all_uids,
                              exprs,
                              joiner)
            return construct_expr(join_ir, new_schema, indices, aggregations)
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

                if not (is_subset_row_key or is_interval):
                    # foreign-key join
                    foreign_key_annotates = {Env.get_uid(): e for e in exprs}

                    # contains original key and join key
                    join_table = src.select_rows(**foreign_key_annotates).rows()

                    join_table = join_table.key_by(*foreign_key_annotates)

                    value_uid = Env.get_uid()
                    join_table = join_table.annotate(**{value_uid: right.index(join_table.key)})

                    #  FIXME: Maybe zip join here?
                    join_table = join_table.group_by(*src.row_key).aggregate(
                        **{uid:
                           hl.dict(hl.agg.collect(hl.tuple([hl.tuple([join_table[f] for f in foreign_key_annotates]),
                                                            join_table[value_uid]])))})

                    def joiner(left: MatrixTable):
                        return MatrixTable(
                            ir.MatrixMapRows(
                                ir.MatrixAnnotateRowsTable(left._mir, join_table._tir, uid),
                                ir.InsertFields(
                                    ir.Ref('va'),
                                    [(uid, ir.Apply('get', join_table._row_type[uid].value_type,
                                                    ir.GetField(ir.GetField(ir.Ref('va'), uid), uid),
                                                    ir.MakeTuple([e._ir for e in exprs])))],
                                    None)))
                else:
                    def joiner(left: MatrixTable):
                        return MatrixTable(ir.MatrixAnnotateRowsTable(left._mir, right._tir, uid, all_matches))
                ast = ir.Join(ir.GetField(ir.TopLevelReference('va'), uid),
                              [uid],
                              exprs,
                              joiner)
                return construct_expr(ast, new_schema, indices, aggregations)
            elif indices == src._col_indices and not (is_interval and all_matches):
                all_uids = [uid]
                if len(exprs) == len(src.col_key) and all([
                        exprs[i] is src.col_key[i] for i in range(len(exprs))]):
                    # key is already correct
                    def joiner(left):
                        return MatrixTable(ir.MatrixAnnotateColsTable(left._mir, right._tir, uid))
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
                        result = MatrixTable(ir.MatrixAnnotateColsTable(
                            (left.add_col_index(index_uid)
                             .key_cols_by(index_uid)
                             ._mir),
                            joined._tir,
                            uid)).key_cols_by(*prev_key)
                        return result
                join_ir = ir.Join(ir.GetField(ir.TopLevelReference('sa'), uid),
                                  all_uids,
                                  exprs,
                                  joiner)
                return construct_expr(join_ir, new_schema, indices, aggregations)
            else:
                raise NotImplementedError()
        else:
            raise TypeError("Cannot join with expressions derived from '{}'".format(src.__class__))

    def index_globals(self) -> 'StructExpression':
        """Return this table's global variables for use in another
        expression context.

        Examples
        --------
        >>> table_result = table2.annotate(C = table2.A * table1.index_globals().global_field_1)

        Returns
        -------
        :class:`.StructExpression`
        """
        return construct_expr(ir.TableGetGlobals(self._tir), self.globals.dtype)

    def _process_joins(self, *exprs) -> 'Table':
        return process_joins(self, exprs)

    def cache(self) -> 'Table':
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
    def persist(self, storage_level='MEMORY_AND_DISK') -> 'Table':
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
        return Env.backend().persist_table(self, storage_level)

    def unpersist(self) -> 'Table':
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
        return Env.backend().unpersist_table(self)

    @typecheck_method(_localize=bool)
    def collect(self, _localize=True):
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
        if len(self.key) > 0:
            t = self.order_by(*self.key)
        else:
            t = self
        rows_ir = ir.GetField(ir.TableCollect(t._tir), 'rows')
        e = construct_expr(rows_ir, hl.tarray(t.row.dtype))
        if _localize:
            return Env.backend().execute(e._ir)
        else:
            return e

    def describe(self, handler=print, *, widget=False):
        """Print information about the fields in the table.

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
    def add_index(self, name='idx') -> 'Table':
        """Add the integer index of each row as a new row field.

        Examples
        --------

        >>> table_result = table1.add_index()
        >>> table_result.show()  # doctest: +SKIP_OUTPUT_CHECK
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

    @typecheck_method(tables=table_type, unify=bool)
    def union(self, *tables, unify: bool = False) -> 'Table':
        """Union the rows of multiple tables.

        Examples
        --------

        Take the union of rows from two tables:

        >>> union_table = table1.union(other_table)

        Notes
        -----
        If a row appears in more than one table identically, it is duplicated
        in the result. All tables must have the same key names and types. They
        must also have the same row types, unless the `unify` parameter is
        ``True``, in which case a field appearing in any table will be included
        in the result, with missing values for tables that do not contain the
        field. If a field appears in multiple tables with incompatible types,
        like arrays and strings, then an error will be raised.

        Parameters
        ----------
        tables : varargs of :class:`.Table`
            Tables to union.
        unify : :obj:`bool`
            Attempt to unify table field.

        Returns
        -------
        :class:`.Table`
            Table with all rows from each component table.
        """
        left_key = self.key.dtype
        for i, ht, in enumerate(tables):
            if left_key != ht.key.dtype:
                raise ValueError(f"'union': table {i} has a different key."
                                 f"  Expected:  {left_key}\n"
                                 f"  Table {i}: {ht.key.dtype}")

            if not (unify or ht.row.dtype == self.row.dtype):
                raise ValueError(f"'union': table {i} has a different row type.\n"
                                 f"  Expected:  {self.row.dtype}\n"
                                 f"  Table {i}: {ht.row.dtype}\n"
                                 f"  If the tables have the same fields in different orders, or some\n"
                                 f"    common and some unique fields, then the 'unify' parameter may be\n"
                                 f"    able to coerce the tables to a common type.")
        all_tables = [self]
        all_tables.extend(tables)

        if unify and not len(set(ht.row_value.dtype for ht in all_tables)) == 1:
            discovered = collections.defaultdict(dict)
            for i, ht in enumerate(all_tables):
                for field_name in ht.row_value:
                    discovered[field_name][i] = ht[field_name]
            all_fields = [{} for _ in all_tables]
            for field_name, expr_dict in discovered.items():
                *unified, can_unify = hl.expr.expressions.unify_exprs(*expr_dict.values())
                if not can_unify:
                    raise ValueError(f"cannot unify field {field_name!r}: found fields of types "
                                     f"{[str(t) for t in {e.dtype for e in expr_dict.values()}]}")
                unified_map = dict(zip(expr_dict.keys(), unified))
                default = hl.missing(unified[0].dtype)
                for i in range(len(all_tables)):
                    all_fields[i][field_name] = unified_map.get(i, default)

            for i, t in enumerate(all_tables):
                all_tables[i] = t.select(**all_fields[i])

        return Table(ir.TableUnion([table._tir for table in all_tables]))

    @typecheck_method(n=int, _localize=bool)
    def take(self, n, _localize=True):
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

        return self.head(n).collect(_localize)

    @typecheck_method(n=int)
    def head(self, n) -> 'Table':
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

        return Table(ir.TableHead(self._tir, n))

    @typecheck_method(n=int)
    def tail(self, n) -> 'Table':
        """Subset table to last `n` rows.

        Examples
        --------
        Subset to the last three rows:

        >>> table_result = table1.tail(3)
        >>> table_result.count()
        3

        Notes
        -----

        The number of partitions in the new table is equal to the number of
        partitions containing the last `n` rows.

        Parameters
        ----------
        n : int
            Number of rows to include.

        Returns
        -------
        :class:`.Table`
            Table including the last `n` rows.
        """

        return Table(ir.TableTail(self._tir, n))

    @typecheck_method(p=numeric,
                      seed=nullable(int))
    def sample(self, p, seed=None) -> 'Table':
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

        if not 0 <= p <= 1:
            raise ValueError("Requires 'p' in [0,1]. Found p={}".format(p))

        return self.filter(hl.rand_bool(p, seed))

    @typecheck_method(n=int,
                      shuffle=bool)
    def repartition(self, n, shuffle=True) -> 'Table':
        """Change the number of partitions.

        Examples
        --------

        Repartition to 500 partitions:

        >>> table_result = table1.repartition(500)

        Notes
        -----

        Check the current number of partitions with :meth:`.n_partitions`.

        The data in a dataset is divided into chunks called partitions, which
        may be stored together or across a network, so that each partition may
        be read and processed in parallel by available cores. When a table with
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
        Hail combines existing partitions to avoid a full shuffle.
        These algorithms correspond to the `repartition` and
        `coalesce` commands in Spark, respectively. In particular,
        when ``shuffle=False``, ``n_partitions`` cannot exceed current
        number of partitions.

        Parameters
        ----------
        n : int
            Desired number of partitions.
        shuffle : bool
            If ``True``, use full shuffle to repartition.

        Returns
        -------
        :class:`.Table`
            Repartitioned table.
        """

        return Table(ir.TableRepartition(
            self._tir, n, ir.RepartitionStrategy.SHUFFLE if shuffle else ir.RepartitionStrategy.COALESCE))

    @typecheck_method(max_partitions=int)
    def naive_coalesce(self, max_partitions: int) -> 'Table':
        """Naively decrease the number of partitions.

        Example
        -------
        Naively repartition to 10 partitions:

        >>> table_result = table1.naive_coalesce(10)

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
        :class:`.Table`
            Table with at most `max_partitions` partitions.
        """

        return Table(ir.TableRepartition(
            self._tir, max_partitions, ir.RepartitionStrategy.NAIVE_COALESCE))

    @typecheck_method(other=table_type)
    def semi_join(self, other: 'Table') -> 'Table':
        """Filters the table to rows whose key appears in `other`.

        Parameters
        ----------
        other : :class:`.Table`
            Table with compatible key field(s).

        Returns
        -------
        :class:`.Table`

        Notes
        -----
        The key type of the table must match the key type of `other`.

        This method does not change the schema of the table; it is a method of
        filtering the table to keys present in another table.

        To discard keys present in `other`, use :meth:`.anti_join`.

        Examples
        --------
        >>> table_result = table1.semi_join(table2)

        It may be expensive to key the left-side table by the right-side key.
        In this case, it is possible to implement a semi-join using a non-key
        field as follows:

        >>> table_result = table1.filter(hl.is_defined(table2.index(table1['ID'])))

        See Also
        --------
        :meth:`.anti_join`
        """
        return self.filter(hl.is_defined(other.index(self.key)))

    @typecheck_method(other=table_type)
    def anti_join(self, other: 'Table') -> 'Table':
        """Filters the table to rows whose key does not appear in `other`.

        Parameters
        ----------
        other : :class:`.Table`
            Table with compatible key field(s).

        Returns
        -------
        :class:`.Table`

        Notes
        -----
        The key type of the table must match the key type of `other`.

        This method does not change the schema of the table; it is a method of
        filtering the table to keys not present in another table.

        To restrict to keys present in `other`, use :meth:`.semi_join`.

        Examples
        --------
        >>> table_result = table1.anti_join(table2)

        It may be expensive to key the left-side table by the right-side key.
        In this case, it is possible to implement an anti-join using a non-key
        field as follows:

        >>> table_result = table1.filter(hl.is_missing(table2.index(table1['ID'])))

        See Also
        --------
        :meth:`.semi_join`, :meth:`.filter`
        """
        return self.filter(hl.is_missing(other.index(self.key)))

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
        Tables are joined at rows whose key fields have equal values. Missing values never match.
        The inclusion of a row with no match in the opposite table depends on the
        join type:

        - **inner** -- Only rows with a matching key in the opposite table are included
          in the resulting table.
        - **left** -- All rows from the left table are included in the resulting table.
          If a row in the left table has no match in the right table, then the fields
          derived from the right table will be missing.
        - **right** -- All rows from the right table are included in the resulting table.
          If a row in the right table has no match in the left table, then the fields
          derived from the left table will be missing.
        - **outer** -- All rows are included in the resulting table. If a row in the right
          table has no match in the left table, then the fields derived from the left
          table will be missing. If a row in the right table has no match in the left table,
          then the fields derived from the left table will be missing.

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
            Table to join.
        how : :class:`str`
            Join type. One of "inner", "left", "right", "outer"

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
        left_fields = set(self._fields)
        right_fields = set(right._fields)

        renames, _ = deduplicate(
            right_fields, max_attempts=100, already_used=left_fields)

        if renames:
            renames = dict(renames)
            right = right.rename(renames)
            info('Table.join: renamed the following fields on the right to avoid name conflicts:'
                 + ''.join(f'\n    {repr(k)} -> {repr(v)}' for k, v in renames.items()))

        return Table(ir.TableJoin(self._tir, right._tir, how, len(self.key)))

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
        return self.aggregate(hl.agg.all(expr))

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
        return self.aggregate(hl.agg.any(expr))

    @typecheck_method(mapping=dictof(str, str))
    def rename(self, mapping) -> 'Table':
        """Rename fields of the table.

        Examples
        --------
        Rename `C1` to `col1` and `C2` to `col2`:

        >>> table_result = table1.rename({'C1' : 'col1', 'C2' : 'col2'})

        Parameters
        ----------
        mapping : :obj:`dict` of :class:`str`, :obj:`str`
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

        return Table(ir.TableRename(self._tir, row_map, global_map))

    def expand_types(self) -> 'Table':
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

        t = self
        if len(t.key) > 0:
            t = t.order_by(*t.key)

        def _expand(e):
            if isinstance(e, CollectionExpression) or isinstance(e, DictExpression):
                return hl.map(lambda x: _expand(x), hl.array(e))
            elif isinstance(e, StructExpression):
                return hl.struct(**{k: _expand(v) for (k, v) in e.items()})
            elif isinstance(e, TupleExpression):
                return hl.struct(**{f'_{i}': x for (i, x) in enumerate(e)})
            elif isinstance(e, IntervalExpression):
                return hl.struct(start=e.start,
                                 end=e.end,
                                 includesStart=e.includes_start,
                                 includesEnd=e.includes_end)
            elif isinstance(e, LocusExpression):
                return hl.struct(contig=e.contig,
                                 position=e.position)
            elif isinstance(e, CallExpression):
                return hl.struct(alleles=hl.map(lambda i: e[i], hl.range(0, e.ploidy)),
                                 phased=e.phased)
            elif isinstance(e, NDArrayExpression):
                return hl.struct(shape=e.shape, data=_expand(e._data_array()))
            else:
                assert isinstance(e, (NumericExpression, BooleanExpression, StringExpression))
                return e

        t = t.select(**_expand(t.row))
        t = t.select_globals(**_expand(t.globals))
        return t

    def flatten(self) -> 'Table':
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
        # unkey but preserve order
        t = self.order_by(*self.key)
        t = Table(ir.TableMapRows(t._tir, t.row.flatten()._ir))
        return t

    @typecheck_method(exprs=oneof(str, Expression, Ascending, Descending))
    def order_by(self, *exprs) -> 'Table':
        """Sort by the specified fields, defaulting to ascending order. Will unkey the table if it is keyed.

        Examples
        --------
        Let's assume we have a field called `HT` in our table.

        By default, ascending order is used:

        >>> sorted_table = table1.order_by(table1.HT)

        >>> sorted_table = table1.order_by('HT')

        You can sort in ascending order explicitly:

        >>> sorted_table = table1.order_by(hl.asc(table1.HT))

        >>> sorted_table = table1.order_by(hl.asc('HT'))

        Tables can be sorted by field descending order as well:

        >>> sorted_table = table1.order_by(hl.desc(table1.HT))

        >>> sorted_table = table1.order_by(hl.desc('HT'))

        Tables can also be sorted on multiple fields:

        >>> sorted_table = table1.order_by(hl.desc('HT'), hl.asc('SEX'))

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
        exprs : varargs of :func:`~.asc`, :func:`.desc`, :class:`.Expression`, or :class:`str`
            Fields to sort by.

        Returns
        -------
        :class:`.Table`
            Table sorted by the given fields.
        """
        lifted_exprs = []
        for e in exprs:
            sort_type = 'A'
            if isinstance(e, Ascending):
                e = e.col
            elif isinstance(e, Descending):
                e = e.col
                sort_type = 'D'

            if isinstance(e, str):
                expr = self[e]
            else:
                expr = e
            lifted_exprs.append((expr, sort_type))

        sort_fields = []
        complex_exprs = {}

        for e, sort_type in lifted_exprs:
            if e._indices.source is not self:
                if e._indices.source is None:
                    raise ValueError("Sort fields must be fields of the callee Table, found scalar expression")
                else:
                    raise ValueError(f"Sort fields must be fields of the callee Table,"
                                     f" found field of {e._indices.source}")
            elif e._indices != self._row_indices:
                raise ValueError("Sort fields must be row-indexed, found global sort expression")
            else:
                field_name = self._fields_inverse.get(e)
                if field_name is None:
                    field_name = Env.get_uid()
                    complex_exprs[field_name] = e
                sort_fields.append((field_name, sort_type))

        t = self
        if complex_exprs:
            t = t.annotate(**complex_exprs)
        t = Table(ir.TableOrderBy(t._tir, sort_fields))
        if complex_exprs:
            t = t.drop(*complex_exprs.keys())
        return t

    @typecheck_method(field=oneof(str, Expression),
                      name=nullable(str))
    def explode(self, field, name=None) -> 'Table':
        """Explode rows along a field of type array or set, copying the entire row for each element.

        Examples
        --------
        `people_table` is a :class:`.Table` with three fields: `Name`, `Age`
        and `Children`.

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
        >>> exploded.show() # doctest: +SKIP_OUTPUT_CHECK
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
        >>> exploded.show() # doctest: +SKIP_OUTPUT_CHECK
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
        Each row is copied for each element of `field`. The explode operation
        unpacks the elements in a field of type ``array`` or ``set`` into its
        own row. If an empty ``array`` or ``set`` is exploded, the entire row is
        removed from the table. In the example above, notice that the name
        "Caroline" is not found in the exploded table.

        Missing arrays or sets are treated as empty.

        Currently, the `name` argument may not be used if `field` is not a
        top-level field of the table (e.g. `name` may be used with ``ht.foo``
        but not ``ht.foo.bar``).

        Parameters
        ----------
        field : :class:`str` or :class:`.Expression`
            Top-level field name or expression.
        name : :class:`str` or None
            If not `None`, rename the exploded field to `name`.

        Returns
        -------
        :class:`.Table`
        """
        if isinstance(field, str):
            if field not in self._fields:
                raise KeyError("Table has no field '{}'".format(field))
            elif self._fields[field]._indices != self._row_indices:
                raise ExpressionException("Method 'explode' expects a field indexed by row, found axes '{}'"
                                          .format(self._fields[field]._indices.axes))
            root = [field]
            field = self._fields[field]
        else:
            analyze('Table.explode', field, self._row_indices, set(self._fields.keys()))
            if not field._ir.is_nested_field:
                raise ExpressionException(
                    "method 'explode' requires a field or subfield, not a complex expression")
            nested = field._ir
            root = []
            while isinstance(nested, ir.GetField):
                root.append(nested.name)
                nested = nested.o
            root = root[::-1]

        if not isinstance(field.dtype, (tarray, tset)):
            raise ValueError(f"method 'explode' expects array or set, found: {field.dtype}")

        for k in self.key.values():
            if k is field:
                raise ValueError("method 'explode' cannot explode a key field")

        t = Table(ir.TableExplode(self._tir, root))
        if name is not None:
            if len(root) > 1:
                raise ValueError("'Table.explode' does not support the 'name' argument when exploding nested fields")
            t = t.rename({root[0]: name})
        return t

    @typecheck_method(row_key=sequenceof(str),
                      col_key=sequenceof(str),
                      row_fields=sequenceof(str),
                      col_fields=sequenceof(str),
                      n_partitions=nullable(int))
    def to_matrix_table(self, row_key, col_key, row_fields=[], col_fields=[], n_partitions=None) -> 'hl.MatrixTable':
        """Construct a matrix table from a table in coordinate representation.

        Examples
        --------
        Import a coordinate-representation table from disk:

        >>> coord_ht = hl.import_table('data/coordinate_matrix.tsv', impute=True)
        >>> coord_ht.show()
        +---------+---------+----------+
        | row_idx | col_idx |        x |
        +---------+---------+----------+
        |   int32 |   int32 |  float64 |
        +---------+---------+----------+
        |       1 |       1 | 2.50e-01 |
        |       1 |       2 | 3.30e-01 |
        |       2 |       1 | 1.10e-01 |
        |       3 |       1 | 1.00e+00 |
        |       3 |       2 | 0.00e+00 |
        +---------+---------+----------+

        Convert to a matrix table and show:

        >>> dense_mt = coord_ht.to_matrix_table(row_key=['row_idx'], col_key=['col_idx'])
        >>> dense_mt.show()
        +---------+----------+----------+
        | row_idx |      1.x |      2.x |
        +---------+----------+----------+
        |   int32 |  float64 |  float64 |
        +---------+----------+----------+
        |       1 | 2.50e-01 | 3.30e-01 |
        |       2 | 1.10e-01 |       NA |
        |       3 | 1.00e+00 | 0.00e+00 |
        +---------+----------+----------+

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
        c = collections.Counter(all_fields)
        row_field_set = set(self.row)
        for k, v in c.items():
            if k not in row_field_set:
                raise ValueError(f"'to_matrix_table': field {repr(k)} is not a row field")
            if v > 1:
                raise ValueError(f"'to_matrix_table': field {repr(k)} appeared in {v} field groups")

        if len(row_key) == 0:
            raise ValueError("'to_matrix_table': require at least one row key field")
        if len(col_key) == 0:
            raise ValueError("'to_matrix_table': require at least one col key field")

        ht = self.key_by()

        non_entry_fields = set(itertools.chain(row_key, col_key, row_fields, col_fields))
        entry_fields = [x for x in ht.row if x not in non_entry_fields]

        if not entry_fields:
            raise ValueError("'Table.to_matrix_table': no fields remain as entry fields:\n"
                             "  all table fields found in one of 'row_key', 'col_key', 'row_fields', 'col_fields'")

        col_data = hl.rbind(
            hl.array(
                ht.aggregate(
                    hl.agg.group_by(ht.row.select(*col_key), hl.agg.take(ht.row.select(*col_fields), 1)[0]),
                    _localize=False)),
            lambda data: hl.struct(data=data,
                                   key_to_index=hl.dict(hl.range(0, hl.len(data)).map(lambda i: (data[i][0], i))))
        )

        col_data_uid = Env.get_uid()
        ht = ht.drop(*col_fields)
        ht = ht.annotate_globals(**{col_data_uid: col_data})

        entries_uid = Env.get_uid()
        ht = (ht.group_by(*row_key)
              .partition_hint(n_partitions)
              # FIXME: should be agg._prev_nonnull https://github.com/hail-is/hail/issues/5345
              .aggregate(**{x: hl.agg.take(ht[x], 1)[0] for x in row_fields},
                         **{entries_uid: hl.rbind(
                             hl.dict(hl.agg.collect((ht[col_data_uid]['key_to_index'][ht.row.select(*col_key)],
                                                     ht.row.select(*entry_fields)))),
                             lambda entry_dict: hl.range(0, hl.len(ht[col_data_uid]['key_to_index']))
                             .map(lambda i: entry_dict.get(i)))}))
        ht = ht.annotate_globals(
            **{col_data_uid: hl.array(ht[col_data_uid]['data'].map(lambda elt: hl.struct(**elt[0], **elt[1])))})
        return ht._unlocalize_entries(entries_uid, col_data_uid, col_key)

    @typecheck_method(columns=sequenceof(str), entry_field_name=nullable(str), col_field_name=str)
    def to_matrix_table_row_major(self, columns, entry_field_name=None, col_field_name='col'):
        """Construct a matrix table from a table in row major representation. Each element in `columns`
        is a field that will become an entry field in the matrix table. Fields omitted from `columns` become row
        fields. If `columns` are structs, then the matrix table will have the entry fields of those structs. Otherwise,
        the matrix table will have one entry field named `entry_field_name` whose values come from the values
        of the `columns` fields. The matrix table is column indexed by `col_field_name`.

        If you find yourself using this method after :func:`.import_table`,
        consider instead using :func:`.import_matrix_table`.

        Examples
        --------

        Convert a table of RNA expression samples to a :class:`.MatrixTable`:

        >>> t = hl.import_table('data/rna_expression.tsv', impute=True)
        >>> t = t.key_by('gene')
        >>> t.show()
        +---------+---------+---------+----------+-----------+-----------+-----------+
        | gene    | lung001 | lung002 | heart001 | muscle001 | muscle002 | muscle003 |
        +---------+---------+---------+----------+-----------+-----------+-----------+
        | str     |   int32 |   int32 |    int32 |     int32 |     int32 |     int32 |
        +---------+---------+---------+----------+-----------+-----------+-----------+
        | "LD4"   |       1 |       2 |        0 |         2 |         1 |         1 |
        | "SCN1A" |       2 |       1 |        1 |         0 |         0 |         0 |
        | "TITIN" |       3 |       0 |        0 |         1 |         2 |         1 |
        +---------+---------+---------+----------+-----------+-----------+-----------+
        >>> mt = t.to_matrix_table_row_major(
        ...          columns=['lung001', 'lung002', 'heart001',
        ...                   'muscle001', 'muscle002', 'muscle003'],
        ...          entry_field_name='expression',
        ...          col_field_name='sample')
        >>> mt.describe()
        ----------------------------------------
        Global fields:
            None
        ----------------------------------------
        Column fields:
            'sample': str
        ----------------------------------------
        Row fields:
            'gene': str
        ----------------------------------------
        Entry fields:
            'expression': int32
        ----------------------------------------
        Column key: ['sample']
        Row key: ['gene']
        ----------------------------------------
        >>> mt.show(n_cols=2)
        +---------+----------------------+----------------------+
        | gene    | 'lung001'.expression | 'lung002'.expression |
        +---------+----------------------+----------------------+
        | str     |                int32 |                int32 |
        +---------+----------------------+----------------------+
        | "LD4"   |                    1 |                    2 |
        | "SCN1A" |                    2 |                    1 |
        | "TITIN" |                    3 |                    0 |
        +---------+----------------------+----------------------+
        showing the first 2 of 6 columns

        Notes
        -----
        All fields in `columns` must have the same type.

        Parameters
        ----------
        columns : Sequence[str]
            Fields to be used as columns.
        entry_field_name : :class:`str` or None
            Field name for the entries of the matrix table.
        col_field_name : :class:`str`
            Field name for the columns of the matrix table.

        Returns
        -------
        :class:`.MatrixTable`
        """
        if len(columns) == 0:
            raise ValueError('Columns must be non-empty.')

        fields = [self[field] for field in columns]
        col_types = set([field.dtype for field in fields])
        if len(col_types) != 1:
            raise ValueError('All columns must have the same type.')

        if all([isinstance(col_typ, hl.tstruct) for col_typ in col_types]):
            if entry_field_name is not None:
                raise ValueError('Cannot both provide struct columns and an entry field name.')
            entries = hl.array(fields)
        else:
            if entry_field_name is None:
                raise ValueError('Must provide an entry field name.')
            entries = hl.array([hl.struct(**{entry_field_name: field}) for field in fields])

        t = self.transmute(entries=entries)
        t = t.annotate_globals(cols=hl.array([hl.struct(**{col_field_name: col}) for col in columns]))
        return t._unlocalize_entries('entries', 'cols', [col_field_name])

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
    def from_spark(df, key=[]) -> 'Table':
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

        key : :class:`str` or :obj:`list` of :obj:`str`
            Key fields.

        Returns
        -------
        :class:`.Table`
            Table constructed from the Spark SQL DataFrame.
        """
        return Env.spark_backend('from_spark').from_spark(df, key)

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
        return Env.spark_backend('to_spark').to_spark(self, flatten)

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
        return Env.spark_backend('to_pandas').to_pandas(self, flatten)

    @staticmethod
    @typecheck(df=pandas.DataFrame,
               key=oneof(str, sequenceof(str)))
    def from_pandas(df, key=[]) -> 'Table':
        """Create table from Pandas DataFrame

        Examples
        --------

        >>> t = hl.Table.from_pandas(df) # doctest: +SKIP

        Parameters
        ----------
        df : :class:`.pandas.DataFrame`
            Pandas DataFrame.
        key : :class:`str` or :obj:`list` of :obj:`str`
            Key fields.

        Returns
        -------
        :class:`.Table`
        """
        return Env.spark_backend('from_pandas').from_pandas(df, key)

    @typecheck_method(other=table_type, tolerance=nullable(numeric), absolute=bool)
    def _same(self, other, tolerance=1e-6, absolute=False):
        from hail.expr.functions import _values_similar

        if self._type != other._type:
            print(f'Table._same: types differ: {self._type}, {other._type}')
            return False

        left_global_value = Env.get_uid()
        left_value = Env.get_uid()
        left = self
        left = left.select_globals(**{left_global_value: left.globals})
        left = left.group_by(_key=left.key).aggregate(**{left_value: hl.agg.collect(left.row_value)})

        right_global_value = Env.get_uid()
        right_value = Env.get_uid()
        right = other
        right = right.select_globals(**{right_global_value: right.globals})
        right = right.group_by(_key=right.key).aggregate(**{right_value: hl.agg.collect(right.row_value)})

        t = left.join(right, how='outer')

        if not hl.eval(_values_similar(t[left_global_value], t[right_global_value], tolerance, absolute)):
            g = hl.eval(t.globals)
            print(f'Table._same: globals differ: {g[left_global_value]}, {g[right_global_value]}')
            return False

        if not t.all(hl.is_defined(t[left_value]) & hl.is_defined(t[right_value])
                     & _values_similar(t[left_value], t[right_value], tolerance, absolute)):
            print('Table._same: rows differ:')
            t = t.filter(~ _values_similar(t[left_value], t[right_value], tolerance, absolute))
            bad_rows = t.take(10)
            for r in bad_rows:
                print(f'  Row mismatch at key={r._key}:\n    L: {r[left_value]}\n    R: {r[right_value]}')
            return False

        return True

    def collect_by_key(self, name: str = 'values') -> 'Table':
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
        name : :class:`str`
            Field name for all values per key.

        Returns
        -------
        :class:`.Table`
        """

        import hail.methods.misc as misc
        misc.require_key(self, 'collect_by_key')

        return Table(ir.TableAggregateByKey(
            self._tir,
            hl.struct(**{name: hl.agg.collect(self.row_value)})._ir))

    def distinct(self) -> 'Table':
        """Deduplicate keys, keeping exactly one row for each unique key.

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

        import hail.methods.misc as misc
        misc.require_key(self, 'distinct')

        return Table(ir.TableDistinct(self._tir))

    def summarize(self, handler=None):
        """Compute and print summary information about the fields in the table.

        .. include:: _templates/experimental.rst
        """

        if handler is None:
            handler = hl.utils.default_handler()
        handler(self.row._summarize(top=True))

    @typecheck_method(parts=sequenceof(int), keep=bool)
    def _filter_partitions(self, parts, keep=True) -> 'Table':
        return Table(ir.TableToTableApply(self._tir, {'name': 'TableFilterPartitions', 'parts': parts, 'keep': keep}))

    @typecheck_method(entries_field_name=str,
                      cols_field_name=str,
                      col_key=sequenceof(str))
    def _unlocalize_entries(self, entries_field_name, cols_field_name, col_key) -> 'hl.MatrixTable':
        return hl.MatrixTable(ir.CastTableToMatrix(
            self._tir, entries_field_name, cols_field_name, col_key))

    @staticmethod
    @typecheck(tables=sequenceof(table_type), data_field_name=str, global_field_name=str)
    def multi_way_zip_join(tables, data_field_name, global_field_name) -> 'Table':
        """Combine many tables in a zip join

        .. include:: _templates/experimental.rst

        Notes
        -----
        The row type of the returned table is a struct with the key fields, and
        one extra field, `data_field_name`, which is an array of structs with
        the non key fields, one per input. The array elements are missing if
        their corresponding input had no row with that key or possibly if there
        is another input with more rows with that key than the corresponding
        input.

        The global type of the returned table is an array of structs of the
        global type of all of the inputs.

        The types for every input must be identical, not merely compatible,
        including the keys.

        A zip join is similar to an outer join however rows are not duplicated
        to create the full Cartesian product of duplicate keys. Instead, there
        is exactly one entry in some `data_field_name` array for every row in
        the inputs.

        Parameters
        ----------
        tables : :class:`list` of :class:`Table`
            A list of tables to combine
        data_field_name : :class:`str`
            The name of the resulting data field
        global_field_name : :class:`str`
            The name of the resulting global field

        """
        if not tables:
            raise ValueError('multi_way_zip_join must have at least one table as an argument')
        head = tables[0]
        if any(head.key.dtype != t.key.dtype for t in tables):
            raise TypeError('All input tables to multi_way_zip_join must have the same key type')
        if any(head.row.dtype != t.row.dtype for t in tables):
            raise TypeError('All input tables to multi_way_zip_join must have the same row type')
        if any(head.globals.dtype != t.globals.dtype for t in tables):
            raise TypeError('All input tables to multi_way_zip_join must have the same global type')
        return Table(ir.TableMultiWayZipJoin(
            [t._tir for t in tables], data_field_name, global_field_name))

    def _group_within_partitions(self, name, n):
        def grouping_func(part):
            groups = part.grouped(n)
            key_names = list(self.key)
            return groups.map(lambda group:
                              group[0].select(*key_names, **{name: group}))

        return self._map_partitions(grouping_func)

    @typecheck_method(f=func_spec(1, expr_array(expr_struct())))
    def _map_partitions(self, f):
        rows_uid = 'tmp_rows_' + Env.get_uid()
        globals_uid = 'tmp_globals_' + Env.get_uid()
        expr = construct_expr(ir.ToArray(ir.Ref(rows_uid)), hl.tarray(self.row.dtype), self._row_indices)
        body = f(expr)
        result_t = body.dtype
        if any(k not in result_t.element_type for k in self.key):
            raise ValueError('Table._map_partitions must preserve key fields')

        body_ir = ir.Let('global', ir.Ref(globals_uid), ir.ToStream(body._ir))
        return Table(ir.TableMapPartitions(self._tir, globals_uid, rows_uid, body_ir))

    def _calculate_new_partitions(self, n_partitions):
        """returns a set of range bounds that can be passed to write"""
        return Env.backend().execute(ir.TableToValueApply(
            self._tir,
            {'name': 'TableCalculateNewPartitions',
             'nPartitions': n_partitions}))


table_type.set(Table)
