from __future__ import print_function  # Python 2 and 3 print compatibility

from pyspark.sql import DataFrame

from hail.expr.types import Type, TArray, TStruct
from hail.history import *
from hail.genetics import GenomeReference
from hail.typecheck import *
from hail.utils import wrap_to_list, Struct
from hail.utils.java import *


class Ascending(HistoryMixin):
    @record_init
    def __init__(self, col):
        self._jrep = scala_package_object(Env.hail().table).asc(col)


class Descending(HistoryMixin):
    @record_init
    def __init__(self, col):
        self._jrep = scala_package_object(Env.hail().table).desc(col)


def asc(col):
    """Sort by ``col`` ascending."""

    return Ascending(col)


def desc(col):
    """Sort by ``col`` descending."""

    return Descending(col)


kt_type = lazy()


class KeyTable(HistoryMixin):
    """Hail's version of a SQL table where columns can be designated as keys.

    Key tables may be imported from a text file or Spark DataFrame with :py:meth:`hail.api1.HailContext.import_table`
    or :py:meth:`~hail.KeyTable.from_dataframe`, generated from a variant dataset
    with :py:meth:`~hail.VariantDataset.make_table`, :py:meth:`~hail.VariantDataset.genotypes_table`,
    :py:meth:`~hail.VariantDataset.samples_table`, or :py:meth:`~hail.VariantDataset.variants_table`.

    In the examples below, we have imported two key tables from text files (``kt1`` and ``kt2``).

    >>> kt1 = hc.import_table('data/kt_example1.tsv', impute=True)

    +--+---+---+-+-+----+----+----+
    |ID|HT |SEX|X|Z| C1 | C2 | C3 |
    +==+===+===+=+=+====+====+====+
    |1 |65 |M  |5|4|2	|50  |5   |
    +--+---+---+-+-+----+----+----+
    |2 |72 |M  |6|3|2	|61  |1   |
    +--+---+---+-+-+----+----+----+
    |3 |70 |F  |7|3|10	|81  |-5  |
    +--+---+---+-+-+----+----+----+
    |4 |60 |F  |8|2|11	|90  |-10 |
    +--+---+---+-+-+----+----+----+

    >>> kt2 = hc.import_table('data/kt_example2.tsv', impute=True)

    +---+---+------+
    |ID	|A  |B     |
    +===+===+======+
    |1	|65 |cat   |
    +---+---+------+
    |2	|72 |dog   |
    +---+---+------+
    |3	|70 |mouse |
    +---+---+------+
    |4	|60 |rabbit|
    +---+---+------+

    :ivar hc: Hail Context
    :vartype hc: :class:`hail.api1.HailContext`
    """

    def __init__(self, hc, jkt):
        self.hc = hc
        self._jkt = jkt

        self._schema = None
        self._num_columns = None
        self._key = None
        self._column_names = None

    def __repr__(self):
        return self._jkt.toString()

    @classmethod
    @handle_py4j
    @record_classmethod
    @typecheck_method(rows=oneof(listof(Struct), listof(dictof(strlike, anytype))),
                      schema=TStruct,
                      key=oneof(strlike, listof(strlike)),
                      num_partitions=nullable(integral))
    def parallelize(cls, rows, schema, key=[], num_partitions=None):
        """Construct a key table from a list of rows.

        **Examples**

        >>> rows = [{'a': 5, 'b': 'foo', 'c': False},
        ...         {'a': None, 'b': 'bar', 'c': True},
        ...         {'b': 'baz', 'c': False}]
        >>> schema = TStruct(['a', 'b', 'c'], [TInt32(), TString(), TBoolean()])
        >>> table = KeyTable.parallelize(rows, schema, key='b')

        This table will look like:

        .. code-block:: text

            >>> table.to_dataframe().show()

            +----+---+-----+
            |   a|  b|    c|
            +----+---+-----+
            |   5|foo|false|
            |null|bar| true|
            |null|baz|false|
            +----+---+-----+

        :param rows: List of rows to include in table.
        :type rows: list of :class:`.hail.representation.Struct` or dict

        :param schema: Struct schema of table.
        :type schema: :class:`.hail.expr.TStruct`

        :param key: Key field(s).
        :type key: str or list of str

        :param num_partitions: Number of partitions to generate.
        :type num_partitions: int or None

        :return: Key table parallelized from the given rows.
        :rtype: :class:`.KeyTable`
        """

        return KeyTable(
            Env.hc(),
            Env.hail().table.Table.parallelize(
                Env.hc()._jhc, [schema._convert_to_j(r) for r in rows],
                schema._jtype, wrap_to_list(key), joption(num_partitions)))

    @property
    @handle_py4j
    def num_columns(self):
        """Number of columns.

        >>> kt1.num_columns
        8

        :rtype: int
        """

        if self._num_columns is None:
            self._num_columns = self._jkt.nColumns()
        return self._num_columns

    @property
    @handle_py4j
    def schema(self):
        """Table schema.

        **Examples**

        >>> print(kt1.schema)

        The ``pprint`` module can be used to print the schema in a more human-readable format:

        >>> from pprint import pprint
        >>> pprint(kt1.schema)

        :rtype: :class:`.TStruct`
        """

        if self._schema is None:
            self._schema = Type._from_java(self._jkt.signature())
            assert (isinstance(self._schema, TStruct))
        return self._schema

    @property
    @handle_py4j
    def key(self):
        """List of key columns.

        >>> kt1.key
        [u'ID']

        :rtype: list of str
        """

        if self._key is None:
            self._key = list(self._jkt.key())
        return self._key

    @property
    @handle_py4j
    def columns(self):
        """Names of all columns.

        >>> kt1.columns
        [u'ID', u'HT', u'SEX', u'X', u'Z', u'C1', u'C2', u'C3']

        :rtype: list of str
        """

        if self._column_names is None:
            self._column_names = list(self._jkt.columns())
        return self._column_names

    @handle_py4j
    def count(self):
        """Count the number of rows.

        **Examples**
        
        >>> kt1.count()
        
        :rtype: int
        """

        return self._jkt.count()

    @handle_py4j
    @typecheck_method(other=kt_type)
    def same(self, other):
        """Test whether two key tables are identical.

        **Examples**

        >>> if kt1.same(kt2):
        ...     print("KeyTables are the same!")

        :param other: key table to compare against
        :type other: :class:`.KeyTable` 

        :rtype: bool
        """

        return self._jkt.same(other._jkt)

    @handle_py4j
    @write_history('output', parallel='parallel')
    @typecheck_method(output=strlike,
                      types_file=nullable(strlike),
                      header=bool,
                      parallel=nullable(enumeration('separate_header', 'header_per_shard')))
    def export(self, output, types_file=None, header=True, parallel=None):
        """Export to a TSV file.

        **Examples**

        Rename column names of key table and export to file:

        >>> (kt1.rename({'HT' : 'Height'})
        ...     .export("output/kt1_renamed.tsv"))

        **Notes**

        A text file containing the python code to generate this output file is available at ``<output>.history.txt``.

        :param str output: Output file path.

        :param str types_file: Output path of types file.
        
        :param bool header: Write a header using the column names.

        :param parallel: If 'header_per_shard', return a set of files (one per partition) each with a header rather than serially concatenating these files. If 'separate_header', return a separate header file and
            a set of files (one per partition) without the header. If None, concatenate the header and all partitions into one file.
        :type parallel: str or None
        """

        self._jkt.export(output, types_file, header, Env.hail().utils.ExportType.getExportType(parallel))

    @handle_py4j
    @record_method
    @typecheck_method(expr=strlike,
                      keep=bool)
    def filter(self, expr, keep=True):
        """Filter rows.

        **Examples**

        Keep rows where ``C1`` equals 5:

        >>> kt_result = kt1.filter("C1 == 5")

        Remove rows where ``C1`` equals 10:

        >>> kt_result = kt1.filter("C1 == 10", keep=False)

        **Notes**

        The scope for ``expr`` is all column names in the input :class:`KeyTable`.

        For more information, see the documentation on writing `expressions <../overview.html#expressions>`__
        and using the `Hail Expression Language <exprlang.html>`__.

        .. caution::
           When ``expr`` evaluates to missing, the row will be removed regardless of whether ``keep=True`` or ``keep=False``.

        :param str expr: Boolean filter expression.

        :param bool keep: Keep rows where ``expr`` is true.

        :return: Filtered key table.
        :rtype: :class:`.KeyTable`
        """

        return KeyTable(self.hc, self._jkt.filter(expr, keep))

    @handle_py4j
    @record_method
    @typecheck_method(expr=oneof(strlike, listof(strlike)))
    def annotate(self, expr):
        """Add new columns computed from existing columns.

        **Examples**

        Add new column ``Y`` which is equal to 5 times ``X``:

        >>> kt_result = kt1.annotate("Y = 5 * X")


        **Notes**

        The scope for ``expr`` is all column names in the input :class:`KeyTable`.

        For more information, see the documentation on writing `expressions <../overview.html#expressions>`__
        and using the `Hail Expression Language <exprlang.html>`__.

        :param expr: Annotation expression or multiple annotation expressions.
        :type expr: str or list of str

        :return: Key table with new columns specified by ``expr``.
        :rtype: :class:`.KeyTable`
        """

        if isinstance(expr, list):
            expr = ','.join(expr)

        return KeyTable(self.hc, self._jkt.annotate(expr))

    @handle_py4j
    @record_method
    @typecheck_method(right=kt_type,
                      how=strlike)
    def join(self, right, how='inner'):
        """Join two key tables together.

        **Examples**

        Join ``kt1`` to ``kt2`` to produce ``kt_joined``:

        >>> kt_result = kt1.key_by('ID').join(kt2.key_by('ID'))

        **Notes:**

        Hail supports four types of joins specified by ``how``:

         - **inner** -- Key must be present in both ``kt1`` and ``kt2``.
         - **outer** -- Key present in ``kt1`` or ``kt2``. For keys only in ``kt1``, the value of non-key columns from ``kt2`` is set to missing.
           Likewise, for keys only in ``kt2``, the value of non-key columns from ``kt1`` is set to missing.
         - **left** -- Key present in ``kt1``. For keys only in ``kt1``, the value of non-key columns from ``kt2`` is set to missing.
         - **right** -- Key present in ``kt2``. For keys only in ``kt2``, the value of non-key columns from ``kt1`` is set to missing.

        Both key tables must have the same number of keys and the corresponding types of each key must be the same (order matters), but the key names can be different.
        For example, if ``kt1`` has the key schema ``Struct{("a", Int), ("b", String)}``, ``kt1`` can be merged with a key table that has a key schema equal to
        ``Struct{("b", Int), ("c", String)}`` but cannot be merged to a key table with key schema ``Struct{("b", "String"), ("a", Int)}``. ``kt_joined`` will have the same key names and schema as ``kt1``.

        :param  right: Key table to join
        :type right: :class:`.KeyTable`

        :param str how: Method for joining two tables together. One of "inner", "outer", "left", "right".

        :return: Key table that results from joining this key table with another.
        :rtype: :class:`.KeyTable`
        """

        return KeyTable(self.hc, self._jkt.join(right._jkt, how))

    @handle_py4j
    @record_method
    @typecheck_method(key_expr=oneof(strlike, listof(strlike)),
                      agg_expr=oneof(strlike, listof(strlike)),
                      num_partitions=nullable(integral))
    def aggregate_by_key(self, key_expr, agg_expr, num_partitions=None):
        """Aggregate columns programmatically.

        **Examples**

        Compute mean height by sex:

        >>> kt_ht_by_sex = kt1.aggregate_by_key("SEX = SEX", "MEAN_HT = HT.stats().mean")

        The result of :py:meth:`.aggregate_by_key` is a key table ``kt_ht_by_sex`` with the following data:

        +--------+----------+
        |   SEX  |MEAN_HT   |
        +========+==========+
        |   M    |  68.5    |
        +--------+----------+
        |   F    |   65     |
        +--------+----------+

        **Notes**

        The scope for both ``key_expr`` and ``agg_expr`` is all column names in the input :class:`KeyTable`.

        For more information, see the documentation on writing :ref:`expressions <overview-expressions>`
        and using the `Hail Expression Language <https://hail.is/expr_lang.html>`__

        :param key_expr: Named expression(s) for how to compute the keys of the new key table.
        :type key_expr: str or list of str

        :param agg_expr: Named aggregation expression(s).
        :type agg_expr: str or list of str

        :param num_partitions: Target number of partitions in the resulting table.
        :type num_partitions: int or None

        :return: A new key table with the keys computed from the ``key_expr`` and the remaining columns computed from the ``agg_expr``.
        :rtype: :class:`.KeyTable`
        """

        if isinstance(key_expr, list):
            key_expr = ",".join(key_expr)

        if isinstance(agg_expr, list):
            agg_expr = ", ".join(agg_expr)

        return KeyTable(self.hc, self._jkt.aggregate(key_expr, agg_expr, joption(num_partitions)))

    @handle_py4j
    @typecheck_method(expr=strlike)
    def forall(self, expr):
        """Evaluate whether a boolean expression is true for all rows.

        **Examples**

        Test whether all rows in the key table have the value of ``C1`` equal to 5:

        >>> if kt1.forall("C1 == 5"):
        ...     print("All rows have C1 equal 5.")

        :param str expr: Boolean expression.

        :rtype: bool
        """

        return self._jkt.forall(expr)

    @handle_py4j
    @typecheck_method(expr=strlike)
    def exists(self, expr):
        """Evaluate whether a boolean expression is true for at least one row.

        **Examples**

        Test whether any row in the key table has the value of ``C1`` equal to 5:

        >>> if kt1.exists("C1 == 5"):
        ...     print("At least one row has C1 equal 5.")

        :param str expr: Boolean expression.

        :rtype: bool
        """

        return self._jkt.exists(expr)

    @handle_py4j
    @record_method
    @typecheck_method(column_names=oneof(listof(strlike), dictof(strlike, strlike)))
    def rename(self, column_names):
        """Rename columns of key table.

        ``column_names`` can be either a list of new names or a dict
        mapping old names to new names.  If ``column_names`` is a list,
        its length must be the number of columns in this :py:class:`.KeyTable`.

        **Examples**

        Rename using a list:

        >>> kt2.rename(['newColumn1', 'newColumn2', 'newColumn3'])

        Rename using a dict:

        >>> kt2.rename({'A' : 'C1'})

        :param column_names: list of new column names or a dict mapping old names to new names.
        :type list of str or dict of str: str

        :return: Key table with renamed columns.
        :rtype: :class:`.KeyTable`
        """

        return KeyTable(self.hc, self._jkt.rename(column_names))

    @handle_py4j
    @record_method
    def expand_types(self):
        """Expand types Locus, Interval, AltAllele, Variant, Char,
        Set and Dict.  Char is converted to String.  Set is converted
        to Array.  Dict[K, V] is converted to

        .. code-block:: text

            Array[Struct {
                key: K
                value: V
            }]

        :return: key table with signature containing only types:
          Boolean, Int, Long, Float, Double, Array and Struct
        :rtype: :class:`.KeyTable`
        """

        return KeyTable(self.hc, self._jkt.expandTypes())

    @handle_py4j
    @record_method
    @typecheck_method(key=oneof(strlike, listof(strlike)))
    def key_by(self, key):
        """Change which columns are keys.

        **Examples**

        Assume ``kt`` is a :py:class:`.KeyTable` with three columns: c1, c2 and
        c3 and key c1.

        Change key columns:

        >>> kt_result = kt1.key_by(['C2', 'C3'])

        >>> kt_result = kt1.key_by('C2')

        Set to no keys:

        >>> kt_result = kt1.key_by([])

        :param key: List of columns to be used as keys.
        :type key: str or list of str

        :return: Key table whose key columns are given by ``key``.
        :rtype: :class:`.KeyTable`
        """

        return KeyTable(self.hc, self._jkt.keyBy(wrap_to_list(key)))

    @handle_py4j
    @record_method
    def flatten(self):
        """Flatten nested Structs.  Column names will be concatenated with dot
        (.).

        **Examples**

        Flatten Structs in key table:

        >>> kt_result = kt3.flatten()

        Consider a key table ``kt`` with signature

        .. code-block:: text

            a: Struct {
                p: Int
                q: Double
            }
            b: Int
            c: Struct {
                x: String
                y: Array[Struct {
                z: Map[Int]
                }]
            }

        and a single key column ``a``.  The result of flatten is

        .. code-block:: text

            a.p: Int
            a.q: Double
            b: Int
            c.x: String
            c.y: Array[Struct {
                z: Map[Int]
            }]

        with key columns ``a.p, a.q``.

        Note, structures inside non-struct types will not be
        flattened.

        :return: Key table with no columns of type Struct.
        :rtype: :class:`.KeyTable`
        """

        return KeyTable(self.hc, self._jkt.flatten())

    @handle_py4j
    @record_method
    @typecheck_method(selected_columns=oneof(strlike, listof(strlike)),
                      qualified_name=bool)
    def select(self, selected_columns, qualified_name=False):
        """Select a subset of columns.

        **Examples**

        Assume ``kt1`` is a :py:class:`.KeyTable` with three columns: C1, C2 and
        C3.

        Select/drop columns:

        >>> kt_result = kt1.select('C1')

        Reorder the columns:

        >>> kt_result = kt1.select(['C3', 'C1', 'C2'])

        Drop all columns:

        >>> kt_result = kt1.select([])

        Create a new column computed from existing columns:

        >>> kt_result = kt1.select('C_NEW = C1 + C2 + C3')

        Export variant QC results:

        >>> vds.variant_qc()
        ...    .variants_table()
        ...    .select(['v', 'va.qc.*', '`1-AF` = 1 - va.qc.AF'])
        ...    .export('output/variant_qc.tsv')

        **Notes**

        :py:meth:`~hail.KeyTable.select` creates a new schema as specified by `exprs`.

        Each argument can either be an annotation path `A.B.C` for an existing column in the table, a splatted annotation
        path `A.*`, or an :ref:`annotation expression <overview-expr-add>` `Z = X + 2 * Y`.

        For annotation paths, the column name in the new table will be the field name.
        For example, if the annotation path is `A.B.C`, the column name will be `C` in the output table.
        Use the `qualified_name=True` option to output the full path name as the column name (`A.B.C`).

        For annotation paths with a type of Struct, use the splat character `.*` to add a new column per Field
        to the output table. The column name will be the field name unless `qualified_name=True`.

        For annotation expressions, the left hand side is the new annotation path for the computed result.
        To include "." or other special characters in a path name, enclose the name in backticks.

        :param selected_columns: List of columns to be selected.
        :type: str or list of str

        :param qualified_name: If True, make the column name for annotation identifiers arguments be the full path name. Useful for avoiding naming collisions.
        :type: bool

        :return: Key table with selected columns.
        :rtype: :class:`.KeyTable`
        """

        selected_columns = wrap_to_list(selected_columns)
        return KeyTable(self.hc, self._jkt.select(selected_columns, qualified_name))

    @handle_py4j
    @record_method
    @typecheck_method(column_names=oneof(strlike, listof(strlike)))
    def drop(self, column_names):
        """Drop columns.

        **Examples**

        Assume ``kt1`` is a :py:class:`.KeyTable` with three columns: C1, C2 and
        C3.

        Drop columns:

        >>> kt_result = kt1.drop('C1')

        >>> kt_result = kt1.drop(['C1', 'C2'])

        :param column_names: List of columns to be dropped.
        :type: str or list of str

        :return: Key table with dropped columns.
        :rtype: :class:`.KeyTable`
        """

        return KeyTable(self.hc, self._jkt.drop(wrap_to_list(column_names)))

    @handle_py4j
    @typecheck_method(expand=bool,
                      flatten=bool)
    def to_dataframe(self, expand=True, flatten=True):
        """Converts this key table to a Spark DataFrame.

        :param bool expand: If true, expand_types before converting to
          DataFrame.

        :param bool flatten: If true, flatten before converting to
          DataFrame.  If both are true, flatten is run after expand so
          that expanded types are flattened.

        :rtype: :class:`pyspark.sql.DataFrame`
        """

        jkt = self._jkt
        if expand:
            jkt = jkt.expandTypes()
        if flatten:
            jkt = jkt.flatten()
        return DataFrame(jkt.toDF(self.hc._jsql_context), self.hc._sql_context)

    @handle_py4j
    @typecheck_method(expand=bool,
                      flatten=bool)
    def to_pandas(self, expand=True, flatten=True):
        """Converts this key table into a Pandas DataFrame.

        :param bool expand: If true, expand_types before converting to
          Pandas DataFrame.

        :param bool flatten: If true, flatten before converting to Pandas
          DataFrame.  If both are true, flatten is run after expand so
          that expanded types are flattened.

        :returns: Pandas DataFrame constructed from the key table.
        :rtype: :py:class:`pandas.DataFrame`
        """

        return self.to_dataframe(expand, flatten).toPandas()

    @handle_py4j
    @typecheck_method(mode=strlike)
    def export_mongodb(self, mode='append'):
        """Export to MongoDB

        .. warning::

          :py:meth:`~.export_mongodb` is EXPERIMENTAL.

        """

        (scala_package_object(self.hc._hail.driver)
         .exportMongoDB(self.hc._jsql_context, self._jkt, mode))

    @handle_py4j
    @typecheck_method(zk_host=strlike,
                      collection=strlike,
                      block_size=integral)
    def export_solr(self, zk_host, collection, block_size=100):
        """Export to Solr.
        
        .. warning::

          :py:meth:`~.export_solr` is EXPERIMENTAL.

        """

        self._jkt.exportSolr(zk_host, collection, block_size)

    @handle_py4j
    @typecheck_method(address=strlike,
                      keyspace=strlike,
                      table=strlike,
                      block_size=integral,
                      rate=integral)
    def export_cassandra(self, address, keyspace, table, block_size=100, rate=1000):
        """Export to Cassandra.

        .. warning::

          :py:meth:`~.export_cassandra` is EXPERIMENTAL.

        """

        self._jkt.exportCassandra(address, keyspace, table, block_size, rate)

    @handle_py4j
    @record_method
    @typecheck_method(column_names=oneof(strlike, listof(strlike)))
    def explode(self, column_names):
        """Explode columns of this key table.

        The explode operation unpacks the elements in a column of type ``Array`` or ``Set`` into its own row.
        If an empty ``Array`` or ``Set`` is exploded, the entire row is removed from the :py:class:`.KeyTable`.

        **Examples**

        Assume ``kt3`` is a :py:class:`.KeyTable` with three columns: c1, c2 and
        c3.

        >>> kt3 = hc.import_table('data/kt_example3.tsv', impute=True,
        ...                       types={'c1': TString(), 'c2': TArray(TInt32()), 'c3': TArray(TArray(TInt32()))})

        The types of each column are ``String``, ``Array[Int]``, and ``Array[Array[Int]]`` respectively.
        c1 cannot be exploded because its type is not an ``Array`` or ``Set``.
        c2 can only be exploded once because the type of c2 after the first explode operation is ``Int``.

        +----+----------+----------------+
        | c1 |   c2     |   c3           |
        +====+==========+================+
        |  a | [1,2,NA] |[[3,4], []]     |
        +----+----------+----------------+

        Explode c2:

        >>> kt3.explode('c2')

        +----+-------+-----------------+
        | c1 |   c2  |    c3           |
        +====+=======+=================+
        |  a | 1     | [[3,4], []]     |
        +----+-------+-----------------+
        |  a | 2     | [[3,4], []]     |
        +----+-------+-----------------+

        Explode c2 once and c3 twice:

        >>> kt3.explode(['c2', 'c3', 'c3'])

        +----+-------+-------------+
        | c1 |   c2  |   c3        |
        +====+=======+=============+
        |  a | 1     |3            |
        +----+-------+-------------+
        |  a | 2     |3            |
        +----+-------+-------------+
        |  a | 1     |4            |
        +----+-------+-------------+
        |  a | 2     |4            |
        +----+-------+-------------+

        :param column_names: Column name(s) to be exploded.
        :type column_names: str or list of str
            
        :return: Key table with columns exploded.
        :rtype: :class:`.KeyTable`
        """

        return KeyTable(self.hc, self._jkt.explode(wrap_to_list(column_names)))

    @handle_py4j
    @typecheck_method(exprs=oneof(strlike, listof(strlike)))
    def query_typed(self, exprs):
        """Performs aggregation queries over columns of the table, and returns Python object(s) and types.

        **Examples**

        >>> mean_value, t = kt1.query_typed('C1.stats().mean')

        >>> [hist, counter], [t1, t2] = kt1.query_typed(['HT.hist(50, 80, 10)', 'SEX.counter()'])

        See :py:meth:`.Keytable.query` for more information.

        :param exprs:
        :type exprs: str or list of str

        :rtype: (annotation or list of annotation,  :class:`.Type` or list of :class:`.Type`)
        """

        if isinstance(exprs, list):
            result_list = self._jkt.query(jarray(Env.jvm().java.lang.String, exprs))
            ptypes = [Type._from_java(x._2()) for x in result_list]
            annotations = [ptypes[i]._convert_to_py(result_list[i]._1()) for i in xrange(len(ptypes))]
            return annotations, ptypes

        else:
            result = self._jkt.query(exprs)
            t = Type._from_java(result._2())
            return t._convert_to_py(result._1()), t

    @handle_py4j
    @typecheck_method(exprs=oneof(strlike, listof(strlike)))
    def query(self, exprs):
        """Performs aggregation queries over columns of the table, and returns Python object(s).

        **Examples**

        >>> mean_value = kt1.query('C1.stats().mean')

        >>> [hist, counter] = kt1.query(['HT.hist(50, 80, 10)', 'SEX.counter()'])

        **Notes**

        This method evaluates Hail expressions over the rows of the key table.
        The ``exprs`` argument requires either a single string or a list of
        strings. If a single string was passed, then a single result is
        returned. If a list is passed, a list is returned.


        The namespace of the expressions includes one aggregable for each column
        of the key table. We use the example ``kt1`` here, which contains columns
        ``ID``, ``HT``, ``Sex``, ``X``, ``Z``, ``C1``, ``C2``, and ``C3``. Queries
        in this key table will contain the following namespace:

        - ``ID``: (*Aggregable[Int]*)
        - ``HT``: (*Aggregable[Int]*)
        - ``SEX``: (*Aggregable[String]*)
        - ``X``: (*Aggregable[Int]*)
        - ``Z``: (*Aggregable[Int]*)
        - ``C1``: (*Aggregable[Int]*)
        - ``C2``: (*Aggregable[Int]*)
        - ``C3``: (*Aggregable[Int]*)

        Map and filter expressions on these aggregables have the same additional
        scope, which is all the columns in the key table. In our example, this
        includes:

        - ``ID``: (*Int*)
        - ``HT``: (*Int*)
        - ``SEX``: (*String*)
        - ``X``: (*Int*)
        - ``Z``: (*Int*)
        - ``C1``: (*Int*)
        - ``C2``: (*Int*)
        - ``C3``: (*Int*)

        This scope means that operations like the below are permitted:

        >>> fraction_tall_male = kt1.query('HT.filter(x => SEX == "M").fraction(x => x > 70)')

        >>> ids = kt1.query('ID.filter(x => C2 < C3).collect()')

        :param exprs:
        :type exprs: str or list of str

        :rtype: annotation or list of annotation
        """

        r, t = self.query_typed(exprs)
        return r

    @handle_py4j
    def collect(self):
        """Collect table to a local list.

        **Examples**

        >>> id_to_sex = {row.ID : row.SEX for row in kt1.collect()}

        **Notes**

        This method should be used on very small tables and as a last resort.
        It is very slow to convert distributed Java objects to Python
        (especially serially), and the resulting list may be too large
        to fit in memory on one machine.

        :rtype: list of :py:class:`.hail.representation.Struct`
        """

        return TArray(self.schema)._convert_to_py(self._jkt.collect())

    @handle_py4j
    def _typecheck(self):
        """Check if all values with the schema."""

        self._jkt.typeCheck()

    @handle_py4j
    @write_history('output', is_dir=True)
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
    @record_method
    def cache(self):
        """Mark this key table to be cached in memory.

        :py:meth:`~hail.KeyTable.cache` is the same as :func:`persist("MEMORY_ONLY") <hail.KeyTable.persist>`.

        :rtype: :class:`.KeyTable`

        """
        return KeyTable(self.hc, self._jkt.cache())

    @handle_py4j
    @record_method
    @typecheck_method(storage_level=strlike)
    def persist(self, storage_level="MEMORY_AND_DISK"):
        """Persist this key table to memory and/or disk.

        **Examples**

        Persist the key table to both memory and disk:

        >>> kt = kt.persist() # doctest: +SKIP

        **Notes**

        The :py:meth:`~hail.KeyTable.persist` and :py:meth:`~hail.KeyTable.cache` methods 
        allow you to store the current table on disk or in memory to avoid redundant computation and 
        improve the performance of Hail pipelines.

        :py:meth:`~hail.KeyTable.cache` is an alias for 
        :func:`persist("MEMORY_ONLY") <hail.KeyTable.persist>`.  Most users will want "MEMORY_AND_DISK".
        See the `Spark documentation <http://spark.apache.org/docs/latest/programming-guide.html#rdd-persistence>`__ 
        for a more in-depth discussion of persisting data.

        :param storage_level: Storage level.  One of: NONE, DISK_ONLY,
            DISK_ONLY_2, MEMORY_ONLY, MEMORY_ONLY_2, MEMORY_ONLY_SER,
            MEMORY_ONLY_SER_2, MEMORY_AND_DISK, MEMORY_AND_DISK_2,
            MEMORY_AND_DISK_SER, MEMORY_AND_DISK_SER_2, OFF_HEAP
        
        :rtype: :class:`.KeyTable`
        """

        return KeyTable(self.hc, self._jkt.persist(storage_level))

    @handle_py4j
    def unpersist(self):
        """
        Unpersists this table from memory/disk.
        
        **Notes**
        This function will have no effect on a table that was not previously persisted.
        
        There's nothing stopping you from continuing to use a table that has been unpersisted, but doing so will result in
        all previous steps taken to compute the table being performed again since the table must be recomputed. Only unpersist
        a table when you are done with it.
        """
        self._jkt.unpersist()

    @handle_py4j
    @record_method
    @typecheck_method(cols=tupleof(oneof(strlike, Ascending, Descending)))
    def order_by(self, *cols):
        """Sort by the specified columns.  Missing values are sorted after non-missing values.  Sort by the first column, then the second, etc.

        :param cols: Columns to sort by.
        :type: str or asc(str) or desc(str)

        :return: Key table sorted by ``cols``.
        :rtype: :class:`.KeyTable`
        """

        jsort_columns = [asc(col)._jrep if isinstance(col, str) else col._jrep for col in cols]
        return KeyTable(self.hc,
                        self._jkt.orderBy(jarray(Env.hail().table.SortColumn, jsort_columns)))

    @handle_py4j
    def num_partitions(self):
        """Returns the number of partitions in the key table.
        
        :rtype: int
        """
        return self._jkt.nPartitions()

    @classmethod
    @handle_py4j
    @record_classmethod
    @typecheck_method(path=strlike,
                      reference_genome=nullable(GenomeReference))
    def import_interval_list(cls, path, reference_genome=None):
        """Import an interval list file in the GATK standard format.
        
        >>> intervals = KeyTable.import_interval_list('data/capture_intervals.txt')
        
        **The File Format**

        Hail expects an interval file to contain either three or five fields per
        line in the following formats:

        - ``contig:start-end``
        - ``contig  start  end`` (tab-separated)
        - ``contig  start  end  direction  target`` (tab-separated)

        A file in either of the first two formats produces a key table with one column:
        
         - **interval** (*Interval*), key column
         
        A file in the third format (with a "target" column) produces a key with two columns:
        
         - **interval** (*Interval*), key column
         - **target** (*String*)
         
        .. note::

            ``start`` and ``end`` match positions inclusively, e.g. ``start <= position <= end``.
            :py:meth:`~hail.representation.Interval.parse` is exclusive of the end position.

        .. note::

            Hail uses the following ordering for contigs: 1-22 sorted numerically, then X, Y, MT,
            then alphabetically for any contig not matching the standard human chromosomes.

        .. caution::

            The interval parser for these files does not support the full range of formats supported
            by the python parser :py:meth:`~hail.representation.Interval.parse`.  'k', 'm', 'start', and 'end' are all
            invalid motifs in the ``contig:start-end`` format here.

        
        :param str filename: Path to file.

        :param reference_genome: Reference genome to use. Default is :class:`hail.api1.HailContext.default_reference`.
        :type reference_genome: :class:`.GenomeReference`
        
        :return: Interval-keyed table.
        :rtype: :class:`.KeyTable`
        """

        rg = reference_genome if reference_genome else Env.hc().default_reference
        jkt = Env.hail().table.Table.importIntervalList(Env.hc()._jhc, path, rg._jrep)
        return KeyTable(Env.hc(), jkt)

    @classmethod
    @handle_py4j
    @record_classmethod
    @typecheck_method(path=strlike,
                      reference_genome=nullable(GenomeReference))
    def import_bed(cls, path, reference_genome=None):
        """Import a UCSC .bed file as a key table.

        **Examples**

        Add the variant annotation ``va.cnvRegion: Boolean`` indicating inclusion in at least one 
        interval of the three-column BED file `file1.bed`:

        >>> bed = KeyTable.import_bed('data/file1.bed')
        >>> vds_result = vds.annotate_variants_table(bed, root='va.cnvRegion')

        Add a variant annotation **va.cnvRegion** (*String*) with value given by the fourth column of ``file2.bed``:
        
        >>> bed = KeyTable.import_bed('data/file2.bed')
        >>> vds_result = vds.annotate_variants_table(bed, root='va.cnvID')

        The file formats are

        .. code-block:: text

            $ cat data/file1.bed
            track name="BedTest"
            20    1          14000000
            20    17000000   18000000
            ...

            $ cat file2.bed
            track name="BedTest"
            20    1          14000000  cnv1
            20    17000000   18000000  cnv2
            ...


        **Notes**
        
        The key table produced by this method has one of two possible structures. If the .bed file has only
        three fields (``chrom``, ``chromStart``, and ``chromEnd``), then the produced key table has only one
        column:
        
            - **interval** (*Interval*) - Genomic interval.
         
        If the .bed file has four or more columns, then Hail will store the fourth column in the table:
         
             - **interval** (*Interval*) - Genomic interval.
             - **target** (*String*) - Fourth column of .bed file.
         

        `UCSC bed files <https://genome.ucsc.edu/FAQ/FAQformat.html#format1>`__ can have up to 12 fields, 
        but Hail will only ever look at the first four. Hail ignores header lines in BED files.

        .. caution:: 
        
            UCSC BED files are 0-indexed and end-exclusive. The line "5  100  105" will contain
            locus ``5:105`` but not ``5:100``. Details `here <http://genome.ucsc.edu/blog/the-ucsc-genome-browser-coordinate-counting-systems/>`__.

        :param str path: Path to .bed file.

        :param reference_genome: Reference genome to use. Default is :py:meth:`hail.api1.HailContext.default_reference`.
        :type reference_genome: :class:`.GenomeReference`

        :rtype: :class:`.KeyTable`
        """

        rg = reference_genome if reference_genome else Env.hc().default_reference
        jkt = Env.hail().table.Table.importBED(Env.hc()._jhc, path, rg._jrep)
        return KeyTable(Env.hc(), jkt)

    @classmethod
    @handle_py4j
    @record_classmethod
    @typecheck_method(df=DataFrame,
                      key=oneof(strlike, listof(strlike)))
    def from_dataframe(cls, df, key=[]):
        """Convert Spark SQL DataFrame to key table.

        **Examples**

        >>> kt = KeyTable.from_dataframe(df) # doctest: +SKIP

        **Notes**

        Spark SQL data types are converted to Hail types as follows:

        .. code-block:: text

          BooleanType => Boolean
          IntegerType => Int
          LongType => Long
          FloatType => Float
          DoubleType => Double
          StringType => String
          BinaryType => Binary
          ArrayType => Array
          StructType => Struct

        Unlisted Spark SQL data types are currently unsupported.
        
        :param df: PySpark DataFrame.
        :type df: ``DataFrame``
        
        :param key: Key column(s).
        :type key: str or list of str

        :return: Key table constructed from the Spark SQL DataFrame.
        :rtype: :class:`.KeyTable`
        """

        return KeyTable(Env.hc(), Env.hail().table.Table.fromDF(Env.hc()._jhc, df._jdf, wrap_to_list(key)))

    @handle_py4j
    @record_method
    @typecheck_method(n=integral,
                      shuffle=bool)
    def repartition(self, n, shuffle=True):
        """Change the number of distributed partitions.
        
        .. warning ::

          When `shuffle` is `False`, `repartition` can only decrease the number of partitions and simply combines adjacent partitions to achieve the desired number.  It does not attempt to rebalance and so can produce a heavily unbalanced dataset.  An unbalanced dataset can be inefficient to operate on because the work is not evenly distributed across partitions.
        
        :param int n: Desired number of partitions.

        :param bool shuffle: Whether to shuffle or naively coalesce.
        
        :rtype: :class:`.KeyTable` 
        """

        return KeyTable(self.hc, self._jkt.repartition(n, shuffle))

    @classmethod
    @handle_py4j
    @record_classmethod
    @typecheck_method(path=strlike,
                      quantitative=bool,
                      delimiter=strlike,
                      missing=strlike)
    def import_fam(cls, path, quantitative=False, delimiter='\\\\s+', missing='NA'):
        """Import PLINK .fam file into a key table.

        **Examples**

        Import case-control phenotype data from a tab-separated `PLINK .fam
        <https://www.cog-genomics.org/plink2/formats#fam>`_ file into sample
        annotations:

        >>> fam_kt = KeyTable.import_fam('data/myStudy.fam')

        In Hail, unlike PLINK, the user must *explicitly* distinguish between
        case-control and quantitative phenotypes. Importing a quantitative
        phenotype without ``quantitative=True`` will return an error
        (unless all values happen to be ``0``, ``1``, ``2``, and ``-9``):

        >>> fam_kt = KeyTable.import_fam('data/myStudy.fam', quantitative=True)

        **Columns**

        The column, types, and missing values are shown below.

            - **ID** (*String*) -- Sample ID (key column)
            - **famID** (*String*) -- Family ID (missing = "0")
            - **patID** (*String*) -- Paternal ID (missing = "0")
            - **matID** (*String*) -- Maternal ID (missing = "0")
            - **isFemale** (*Boolean*) -- Sex (missing = "NA", "-9", "0")
        
        One of:
    
            - **isCase** (*Boolean*) -- Case-control phenotype (missing = "0", "-9", non-numeric or the ``missing`` argument, if given.
            - **qPheno** (*Double*) -- Quantitative phenotype (missing = "NA" or the ``missing`` argument, if given.

        :param str path: Path to .fam file.

        :param bool quantitative: If True, .fam phenotype is interpreted as quantitative.

        :param str delimiter: .fam file field delimiter regex.

        :param str missing: The string used to denote missing values.
            For case-control, 0, -9, and non-numeric are also treated
            as missing.

        :return: Key table with information from .fam file.
        :rtype: :class:`.KeyTable`
        """

        hc = Env.hc()
        jkt = Env.hail().table.Table.importFam(hc._jhc, path, quantitative, delimiter, missing)
        return KeyTable(hc, jkt)

    @handle_py4j
    @record_method
    @typecheck_method(kts=tupleof(kt_type))
    def union(self, *kts):
        """Union the rows of multiple tables.

        **Examples**

        Take the union of rows from two tables:

        >>> other = hc.import_table('data/kt_example1.tsv', impute=True)
        >>> union_kt = kt1.union(other)

        **Notes**

        If a row appears in both tables identically, it is duplicated in
        the result. The left and right tables must have the same schema
        and key.

        :param kts: Tables to merge.
        :type kts: args of type :class:`.KeyTable`

        :return: A table with all rows from the left and right tables.
        :rtype: :class:`.KeyTable`
        """

        return KeyTable(self.hc, self._jkt.union([kt._jkt for kt in kts]))
    
    @handle_py4j
    @typecheck_method(n=integral)
    def take(self, n):
        """Take a given number of rows from the head of the table.

        **Examples**

        Take the first ten rows:

        >>> first10 = kt1.take(10)

        **Notes**

        This method does not need to look at all the data, and
        allows for fast queries of the start of the table.

        This method is equivalent to :py:meth:`.KeyTable.head` followed by
        :py:meth:`.KeyTable.collect`.

        :param int n: Number of rows to take.

        :return: Rows from the start of the table.
        :rtype: list of :class:`.~hail.representation.Struct`
        """

        return [self.schema._convert_to_py(r) for r in self._jkt.take(n)]

    @handle_py4j
    @record_method
    @typecheck_method(n=integral)
    def head(self, n):
        """Subset table to first n rows.

        **Examples**

        Perform a query on the first 10 rows:

        >>> first10_c1_mean = kt1.head(10).query('C1.stats().mean')

        Return a list with the first 50 rows (equivalent to :py:meth:`.KeyTable.take`):

        >>> first50_rows = kt1.head(50).collect()

        **Notes**

        The number of partitions in the new table is equal to the number
        of partitions containing the first n rows.

        :param int n: Number of rows to include.

        :return: A table subsetted to the first n rows.
        :rtype: :class:`.KeyTable`
        """

        return KeyTable(self.hc, self._jkt.head(n))

    @handle_py4j
    @record_method
    @typecheck_method(name=strlike)
    def indexed(self, name='index'):
        """Add the numerical index of each row as a new column.

        **Examples**

        >>> ind_kt = kt1.indexed()

        **Notes**

        This method returns a table with a new column whose name is
        given by the ``name`` parameter, with type ``Long``. The value
        of this column is the numerical index of each row, starting
        from 0. Methods that respect ordering (like :py:meth:`.KeyTable.take`
        or :py:meth:`.KeyTable.export` will return rows in order.

        This method is helpful for creating a unique integer index for rows
        of a table, so that more complex types can be encoded as a simple
        number.

        :param str name: Name of index column.

        :return: Table with a new index column.
        :rtype: :class:`.KeyTable`
        """

        return KeyTable(self.hc, self._jkt.indexed(name))

    @handle_py4j
    @typecheck_method(n=integral,
                      truncate_to=nullable(integral),
                      print_types=bool)
    def show(self, n=10, truncate_to=None, print_types=True):
        """Show the first few rows of the table in human-readable format.

        **Examples**

        Show, with default parameters (10 rows, no truncation, and column types):

        >>> kt1.show()

        Truncate long columns to 25 characters and only write 5 rows:

        >>> kt1.show(5, truncate_to=25)

        **Notes**

        If the ``truncate_to`` argument is ``None``, then no truncation will
        occur. This is the default behavior. An integer argument will truncate
        each cell to the specified number of characters.

        :param int n: Number of rows to show.

        :param truncate_to: Truncate columns to the desired number of characters.
        :type truncate_to: int or None

        :param bool print_types: Print a line with column types.
        """
        to_print = self._jkt.showString(n, joption(truncate_to), print_types)
        print(to_print)

    @classmethod
    @handle_py4j
    @record_classmethod
    @typecheck_method(n=integral,
                      num_partitions=nullable(integral))
    def range(cls, n, num_partitions=None):
        """Construct a table of ``n`` rows with values 0 to ``n - 1``.

        **Examples**

        Construct a table with 100 rows:

        >>> range_kt = KeyTable.range(100)

        Construct a table with one million rows and twenty partitions:

        >>> range_kt = KeyTable.range(1000000, num_partitions=20)

        **Notes**

        The resulting table has one column:

         - **index** (*Int*) -- Unique row index from 0 to ``n - 1``

        :param int n: Number of rows.

        :param num_partitions: Number of partitions.
        :type num_partitions: int or None

        :rtype: :class:`.KeyTable`
        """

        return KeyTable(Env.hc(), Env.hail().table.Table.range(Env.hc()._jhc, n, joption(num_partitions)))

    @handle_py4j
    @record_method
    @typecheck_method(i=strlike,j=strlike, tie_breaker=nullable(strlike))
    def maximal_independent_set(self, i, j, tie_breaker=None):
        """Compute a `maximal independent set
        <https://en.wikipedia.org/wiki/Maximal_independent_set>`__ of vertices
        in an undirected graph whose edges are given by this key table.

        **Examples**

        Prune individuals from a dataset until no close relationships remain
        with respect to a PC-Relate measure of kinship.

        >>> vds = hc.import_vcf("data/sample.vcf.bgz")
        >>> related_pairs = vds.pc_relate(2, 0.001).filter("kin > 0.125")
        >>> related_samples = related_pairs.query('i.flatMap(i => [i,j]).collectAsSet()')
        >>> related_samples_to_keep = related_pairs.maximal_independent_set("i", "j")
        >>> related_samples_to_remove = related_samples - set(related_samples_to_keep)
        >>> vds.filter_samples_list(list(related_samples_to_remove))

        Prune individuals from a dataset, prefering to keep cases over controls.

        >>> vds = hc.read("data/example.vds")
        >>> related_pairs = vds.pc_relate(2, 0.001).filter("kin > 0.125")
        >>> related_samples = related_pairs.query('i.flatMap(i => [i,j]).collectAsSet()')
        >>> related_samples_to_keep = (related_pairs
        ...   .key_by("i").join(vds.samples_table()).annotate('iAndCase = { id: i, isCase: sa.isCase }')
        ...   .select(['j', 'iAndCase'])
        ...   .key_by("j").join(vds.samples_table()).annotate('jAndCase = { id: j, isCase: sa.isCase }')
        ...   .select(['iAndCase', 'jAndCase'])
        ...   .maximal_independent_set("iAndCase", "jAndCase",
        ...     'if (l.isCase && !r.isCase) -1 else if (!l.isCase && r.isCase) 1 else 0'))
        >>> related_samples_to_remove = related_samples - {x.id for x in related_samples_to_keep}
        >>> vds.filter_samples_list(list(related_samples_to_remove))

        **Notes**

        The vertex set of the graph is implicitly all the values realized by
        ``i`` and ``j`` on the rows of this key table. Each row of the key table
        corresponds to an undirected edge between the vertices given by
        evaluating ``i`` and ``j`` on that row. An undirected edge may appear
        multiple times in the key table and will not affect the output. Vertices
        with self-edges are removed as they are not independent of themselves.

        The expressions for ``i`` and ``j`` must have the same type.

        This method implements a greedy algorithm which iteratively removes a
        vertex of highest degree until the graph contains no edges.

        ``tie_breaker`` is a Hail expression that defines an ordering on
        nodes. It has two values in scope, ``l`` and ``r``, that refer the two
        nodes being compared. A pair of nodes can be ordered in one of three
        ways, and ``tie_breaker`` must encode the relationship as follows:

         - if ``l < r`` then ``tie_breaker`` evaluates to some negative integer
         - if ``l == r`` then ``tie_breaker`` evaluates to 0
         - if ``l > r`` then ``tie_breaker`` evaluates to some positive integer

        For example, the usual ordering on the integers is defined by: ``l - r``.

        When multiple nodes have the same degree, this algorithm will order the
        nodes according to ``tie_breaker`` and remove the *largest* node.

        :param str i: expression to compute one endpoint.
        :param str j: expression to compute another endpoint.
        :param tie_breaker: Expression used to order nodes with equal degree.

        :return: a list of vertices in a maximal independent set.
        :rtype: list of elements with the same type as ``i`` and ``j``

        """

        return jarray_to_list(self._jkt.maximalIndependentSet(i, j, joption(tie_breaker)))

    @handle_py4j
    @record_method
    @typecheck_method(column=strlike,
                      mangle=bool)
    def ungroup(self, column, mangle=False):
        """Lifts fields of struct columns as distinct top-level columns.

        **Examples**

        ``kt4`` is a :py:class:`.KeyTable` with five columns: A, B, C, D and E.

        >>> kt4 = hc.import_table('data/kt_example4.tsv', impute=True,
        ...                       types={'B': TStruct(['B0', 'B1'], [TBoolean(), TString()]),
        ...                              'D': TStruct(['cat', 'dog'], [TInt32(), TInt32()]),
        ...                              'E': TStruct(['A', 'B'], [TInt32(), TInt32()])})

        The types of each column are ``Int32``, ``Struct``, ``Boolean``, ``Struct`` and ``Struct`` respectively.

        +----+--------------------------+-------+-------------------+--------------+
        | A  |   B                      |   C   | D                 | E            |
        +====+==========================+=======+===================+==============+
        | 15 | {"B0":true,"B1":"hello"} | false | {"cat":5,"dog":7} | {"A":5,"B":7}|
        +----+--------------------------+-------+-------------------+--------------+

        Both B and D can be ungrouped. However, E cannot be ungrouped because the names of its
        fields collide with existing columns in the key table.

        Ungroup B:

        >>> kt4.ungroup('B')

        +----+-------+-------------------+--------------+-------+--------+
        | A  |  C    | D                 | E            |   B0  | B1     |
        +====+=======+===================+==============+=======+========+
        | 15 | false | {"cat":5,"dog":7} | {"A":5,"B":7}| true  | "hello"|
        +----+-------+----------------------------------+-------+--------+

        Ungroup E using `mangle=True` to avoid name conflicts:

        >>> kt4.ungroup('E', mangle=True)

        +----+--------------------------+-------+-------------------+--------------+
        | A  |   B                      |   C   | D                 | E.A   |  E.B |
        +====+==========================+=======+===================+==============+
        | 15 | {"B0":true,"B1":"hello"} | false | {"cat":5,"dog":7} | 5     |  7   |
        +----+--------------------------+-------+-------------------+--------------+

        **Notes**

        The ungrouped columns are always appended to the end of the table.

        :param str column: Names of struct column to ungroup.
        :param bool mangle: Rename ungrouped columns as ``column.subcolumn``

        :return: A table with specified column ungrouped.
        :rtype: :class:`.KeyTable`
        """

        return KeyTable(self.hc, self._jkt.ungroup(column, mangle))

    @handle_py4j
    @record_method
    @typecheck_method(dest=strlike,
                      columns=tupleof(strlike))
    def group(self, dest, *columns):
        """Combines columns into a single struct column.

        **Examples**

        ``kt5`` is a :py:class:`.KeyTable` with three columns: A, B, and C.

        >>> kt5 = hc.import_table('data/kt_example5.tsv', impute=True)

        The types of each column are ``Int32``, ``Boolean`` and ``String`` respectively.

        +----+------+-------+
        | A  |   B  |   C   |
        +====+======+=======+
        | 24 | true | "sun" |
        +----+------+-------+

        Group A and C into a new column X:

        >>> kt5.group('X', 'A', 'C')

        +----+-------------------+
        | B  |   X               |
        +====+===================+
        |true|{"A":24,"C":"sun"} |
        +----+-------------------+

        **Notes**

        The grouped column is always appended to the end of the table.

        :param str dest: Name of column to be constructed.

        :param columns: Names of columns to group.
        :type columns: str or list of str

        :return: A table with specified columns grouped.
        :rtype: :class:`.KeyTable`
        """

        return KeyTable(self.hc, self._jkt.group(dest, list(columns)))

    @handle_py4j
    @record_method
    @typecheck_method(expr=oneof(strlike, listof(strlike)))
    def annotate_global_expr(self, expr):
        """Add global fields with expressions.

        **Example**

        Annotate global with an array of populations:

        >>> kt_result = kt1.annotate_global_expr('pops = ["FIN", "AFR", "EAS", "NFE"]')

        :param expr: Annotation expression
        :type expr: str or list of str

        :rtype: :py:class:`.Keytable`
        """

        if isinstance(expr, list):
            expr = ','.join(expr)

        jkt = self._jkt.annotateGlobalExpr(expr)
        return KeyTable(self.hc, jkt)

    @handle_py4j
    @record_method
    @typecheck_method(name=strlike,
                      annotation=anytype,
                      annotation_type=Type)
    def annotate_global(self, name, annotation, annotation_type):
        """Add global annotations from Python objects.

        **Examples**

        Add populations as a global field:

        >>> kt_result = kt1.annotate_global('pops',
        ...                                 ['EAS', 'AFR', 'EUR', 'SAS', 'AMR'],
        ...                                 TArray(TString()))

        :param str name: Name of global field.

        :param annotation: annotation to add to global

        :param annotation_type: Hail type of annotation
        :type annotation_type: :py:class:`.Type`

        :rtype: :py:class:`.KeyTable`
        """

        annotation_type._typecheck(annotation)

        annotated = self._jkt.annotateGlobal(annotation_type._convert_to_j(annotation), annotation_type._jtype, name)
        assert annotated.globalSignature().typeCheck(annotated.globals()), 'error in java type checking'
        return KeyTable(self.hc, annotated)

    @record_method
    def to_hail2(self):
        import hail2
        return hail2.Table(self.hc, self._jkt)

kt_type.set(KeyTable)
