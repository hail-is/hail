from __future__ import print_function  # Python 2 and 3 print compatibility

from hail.java import *
from hail.expr import Type, TArray, TStruct
from hail.utils import wrap_to_list
from pyspark.sql import DataFrame

def asc(col):
    """Sort by ``col`` ascending."""
    
    return scala_package_object(Env.hail().keytable).asc(col)

def desc(col):
    """Sort by ``col`` descending."""
    
    return scala_package_object(Env.hail().keytable).desc(col)

class KeyTable(object):
    """Hail's version of a SQL table where columns can be designated as keys.

    Key tables may be imported from a text file or Spark DataFrame with :py:meth:`~hail.HailContext.import_keytable`
    or :py:meth:`~hail.HailContext.dataframe_to_keytable`, or generated from an existing variant dataset
    with :py:meth:`~hail.VariantDataset.aggregate_by_key`, :py:meth:`~hail.VariantDataset.make_keytable`,
    :py:meth:`~hail.VariantDataset.samples_keytable`, or :py:meth:`~hail.VariantDataset.variants_keytable`.

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
    :vartype hc: :class:`.HailContext`
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

    @staticmethod
    def from_py(hc, rows_py, schema, key_names=[], npartitions=None):
        return KeyTable(
            hc,
            Env.hail().keytable.KeyTable.parallelize(
                hc._jhc, [schema._convert_to_j(r) for r in rows_py],
                schema._jtype, key_names, joption(npartitions)))

    @property
    def num_columns(self):
        """Number of columns.

        >>> kt1.num_columns
        8

        :rtype: int
        """

        if self._num_columns is None:
            self._num_columns = self._jkt.nFields()
        return self._num_columns

    @property
    def schema(self):
        """Key table schema.

        **Example:**

        Print the key table columns / signatures:

        >>> print(kt1.schema)
        Struct {
            ID: Int,
            HT: Int,
            SEX: String,
            X: Int,
            Z: Int,
            C1: Int,
            C2: Int,
            C3: Int
        }

        :rtype: :class:`.TStruct`
        """

        if self._schema is None:
            self._schema = Type._from_java(self._jkt.signature())
            assert (isinstance(self._schema, TStruct))
        return self._schema

    @property
    def key(self):
        """List of key column names.

        >>> kt1.key
        [u'ID']

        :rtype: list of str
        """

        if self._key is None:
            self._key = list(self._jkt.key())
        return self._key

    @property
    def column_names(self):
        """Names of all columns.

        >>> kt1.column_names
        [u'ID', u'HT', u'SEX', u'X', u'Z', u'C1', u'C2', u'C3']

        :rtype: list of str
        """

        if self._column_names is None:
            self._column_names = list(self._jkt.fieldNames())
        return self._column_names

    @handle_py4j
    def count_rows(self):
        """Number of rows.

        >>> kt1.count_rows()
        4L

        :rtype: long
        """

        return self._jkt.nRows()

    @handle_py4j
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
    def export(self, output, types_file=None, header=True):
        """Export to a TSV file.

        **Examples**

        Rename column names of key table and export to file:

        >>> (kt1.rename({'HT' : 'Height'})
        ...     .export("output/kt1_renamed.tsv"))

        :param str output: Output file path.

        :param str types_file: Output path of types file.
        
        :param bool header: Write a header using the column names.
        """

        self._jkt.export(output, types_file, header)

    @handle_py4j
    def filter(self, condition, keep=True):
        """Filter rows.

        **Examples**

        Keep rows where ``C1`` equals 5:

        >>> kt_result = kt1.filter("C1 == 5")

        Remove rows where ``C1`` equals 10:

        >>> kt_result = kt1.filter("C1 == 10", keep=False)

        **Notes**

        The scope for ``condition`` is all column names in the input :class:`KeyTable`.

        For more information, see the documentation on writing `expressions <../overview.html#expressions>`__
        and using the `Hail Expression Language <exprlang.html>`__.

        .. caution::
           When ``condition`` evaluates to missing, the row will be removed regardless of whether ``keep=True`` or ``keep=False``.

        :param str condition: Annotation expression.

        :param bool keep: Keep rows where ``condition`` evaluates to True.

        :return: Key table whose rows have been filtered by evaluating ``condition``.
        :rtype: :class:`.KeyTable`
        """

        return KeyTable(self.hc, self._jkt.filter(condition, keep))

    @handle_py4j
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

        The non-key fields in ``kt2`` must have non-overlapping column names with ``kt1``.

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
    def aggregate_by_key(self, key_expr, agg_expr):
        """Group by key condition and aggregate results.

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

        :return: A new key table with the keys computed from the ``key_expr`` and the remaining columns computed from the ``agg_expr``.
        :rtype: :class:`.KeyTable`
        """

        if isinstance(key_expr, list):
            key_expr = ",".join(key_expr)

        if isinstance(agg_expr, list):
            agg_expr = ", ".join(agg_expr)

        return KeyTable(self.hc, self._jkt.aggregate(key_expr, agg_expr))

    @handle_py4j
    def forall(self, code):
        """Test whether a condition is true for all rows.

        **Examples**

        Test whether all rows in the key table have the value of ``C1`` equal to 5:

        >>> if kt1.forall("C1 == 5"):
        ...     print("All rows have C1 equal 5.")

        :param str code: Boolean expression.

        :rtype: bool
        """

        return self._jkt.forall(code)

    @handle_py4j
    def exists(self, code):
        """Test whether a condition is true for at least one row.

        **Examples**

        Test whether any row in the key table has the value of ``C1`` equal to 5:

        >>> if kt1.exists("C1 == 5"):
        ...     print("At least one row has C1 equal 5.")

        :param str code: Boolean expression.

        :rtype: bool
        """

        return self._jkt.exists(code)

    @handle_py4j
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
    def expand_types(self):
        """Expand types Locus, Interval, AltAllele, Variant, Genotype, Char,
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

        if isinstance(key, list):
            for k in key:
                if not isinstance(k, str) and not isinstance(k, unicode):
                    raise TypeError("expected str or unicode elements of 'key' list, but found %s" % type(k))
        elif not isinstance(key, str) and not isinstance(key, unicode):
            raise TypeError("expected str or list of str for parameter 'key', but found %s" % type(key))

        if not isinstance(key, list):
            key = [key]

        return KeyTable(self.hc, self._jkt.keyBy(key))

    @handle_py4j
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
    def select(self, column_names):
        """Select a subset of columns.

        **Examples**

        Assume ``kt`` is a :py:class:`.KeyTable` with three columns: C1, C2 and
        C3.

        Select/drop columns:

        >>> kt_result = kt1.select(['C1'])

        Reorder the columns:

        >>> kt_result = kt1.select(['C3', 'C1', 'C2'])

        Drop all columns:

        >>> kt_result = kt1.select([])

        :param column_names: List of columns to be selected.
        :type: list of str

        :return: Key table with selected columns.
        :rtype: :class:`.KeyTable`
        """

        new_key = [k for k in self.key if k in column_names]
        return KeyTable(self.hc, self._jkt.select(column_names, new_key))

    @handle_py4j
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
    def export_mongodb(self, mode='append'):
        """Export to MongoDB

        .. warning::

          :py:meth:`~.export_mongodb` is EXPERIMENTAL.

        """
        
        (scala_package_object(self.hc._hail.driver)
         .exportMongoDB(self.hc._jsql_context, self._jkt, mode))

    @handle_py4j
    def export_solr(self, zk_host, collection, block_size=100):
        """Export to Solr.
        
        .. warning::

          :py:meth:`~.export_solr` is EXPERIMENTAL.

        """

        self._jkt.exportSolr(zk_host, collection, block_size)

    @handle_py4j
    def export_cassandra(self, address, keyspace, table, block_size=100, rate=1000):
        """Export to Cassandra.

        .. warning::

          :py:meth:`~.export_cassandra` is EXPERIMENTAL.

        """
        
        self._jkt.exportCassandra(address, keyspace, table, block_size, rate)

    @handle_py4j
    def explode(self, column_names):
        """Explode columns of this key table.

        The explode operation unpacks the elements in a column of type ``Array`` or ``Set`` into its own row.
        If an empty ``Array`` or ``Set`` is exploded, the entire row is removed from the :py:class:`.KeyTable`.

        **Examples**

        Assume ``kt3`` is a :py:class:`.KeyTable` with three columns: c1, c2 and
        c3.

        >>> kt3 = hc.import_table('data/kt_example3.tsv', impute=True,
        ...                          types={'c1': TString(), 'c2': TArray(TInt()), 'c3': TArray(TArray(TInt()))})

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

        if isinstance(column_names, str):
            column_names = [column_names]
        return KeyTable(self.hc, self._jkt.explode(column_names))

    @handle_py4j
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
        """Collect key table as a Python object."""

        return TArray(self.schema)._convert_to_py(self._jkt.collect())

    @handle_py4j
    def _typecheck(self):
        """Check if all values with the schema."""

        self._jkt.typeCheck()

    @handle_py4j
    def write(self, output, overwrite=False):
        """Write as KT file.

        ***Examples***

        >>> kt1.write('output/kt1.kt')

        .. note:: The write path must end in ".kt".       

        :param str output: Path of KT file to write.

        :param bool overwrite: If True, overwrite any existing KT file. Cannot be used 
               to read from and write to the same path.

        """

        self._jkt.write(output, overwrite)

    def cache(self):
        """Mark this key table to be cached in memory.

        :py:meth:`~hail.KeyTable.cache` is the same as :func:`persist("MEMORY_ONLY") <hail.KeyTable.persist>`.

        :rtype: :class:`.KeyTable`

        """
        return KeyTable(self.hc, self._jkt.cache())

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

    def order_by(self, *cols):
        """Sort by the specified columns.  Missing values are sorted after non-missing values.  Sort by the first column, then the second, etc.

        :param cols: Columns to sort by.
        :type: str or asc(str) or desc(str)

        :return: Key table sorted by ``cols``.
        :rtype: :class:`.KeyTable`
        """
        
        jsort_columns = [asc(col) if isinstance(col, str) else col for col in cols]
        return KeyTable(self.hc,
                        self._jkt.orderBy(jarray(Env.hail().keytable.SortColumn, jsort_columns)))

    def num_partitions(self):
        """Returns the number of partitions in the key table.
        
        :rtype: int
        """
        return self._jkt.nPartitions()

    @staticmethod
    @handle_py4j
    def import_interval_list(path):
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
        
        :return: Interval-keyed table.
        :rtype: :class:`.KeyTable`
        """
        jkt = Env.hail().keytable.KeyTable.importIntervalList(Env.hc()._jhc, path)
        return KeyTable(Env.hc(), jkt)

    @staticmethod
    @handle_py4j
    def import_bed(path):
        """Import a UCSC .bed file as a key table.

        **Examples**

        Add the variant annotation ``va.cnvRegion: Boolean`` indicating inclusion in at least one 
        interval of the three-column BED file `file1.bed`:

        >>> bed = KeyTable.import_bed('data/file1.bed')
        >>> vds_result = vds.annotate_variants_table(bed, root='va.cnvRegion')

        Add a variant annotation ``va.cnvRegion: String`` with value given by the fourth column of `file2.bed`:
        
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
         
        If the .bed file has four or more columns, then Hail will store the fourth column as another key 
        table column:
         
         - **interval** (*Interval*) - Genomic interval.
         - **target** (*String*) - Fourth column of .bed file.
         

        `UCSC bed files <https://genome.ucsc.edu/FAQ/FAQformat.html#format1>`_ can have up to 12 fields, 
        but Hail will only ever look at the first four. Hail ignores header lines in BED files.

        .. caution:: UCSC BED files are 0-indexed and end-exclusive. The line "5  100  105" will contain
        locus ``5:105`` but not ``5:100``. Details `here <http://genome.ucsc.edu/blog/the-ucsc-genome-browser-coordinate-counting-systems/>`_.

        :param str path: Path to .bed file.

        :rtype: :class:`.KeyTable`
        """

        jkt = Env.hail().keytable.KeyTable.importBED(Env.hc()._jhc, path)
        return KeyTable(Env.hc(), jkt)

    @staticmethod
    @handle_py4j
    def from_dataframe(df, key=[]):
        """Convert Spark SQL DataFrame to key table.

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

        return KeyTable(Env.hc(), Env.hail().keytable.KeyTable.fromDF(Env.hc()._jhc, df._jdf, wrap_to_list(key)))

    def repartition(self, n):
        """Change the number of distributed partitions.
        
        Always shuffles data.
        
        :param int n: Desired number of partitions.
        
        :rtype: :class:`.KeyTable` 
        """

        return KeyTable(self.hc, self._jkt.repartition(n))

    @staticmethod
    @handle_py4j
    def import_fam(fam_file, quantitative=False, delimiter='\\\\s+', root='sa.fam', missing='NA'):
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

        :param str input: Path to .fam file.

        :param bool quantitative: If True, .fam phenotype is interpreted as quantitative.

        :param str delimiter: .fam file field delimiter regex.

        :param str missing: The string used to denote missing values.
            For case-control, 0, -9, and non-numeric are also treated
            as missing.

        :return: Key table with information from .fam file.
        :rtype: :class:`.KeyTable`
        """

        hc = Env.hc()
        jkt = scala_object(Env.hail().keytable, 'KeyTable').importFam(hc._jhc, input, quantitative, delimiter, missing)
        return KeyTable(hc, jkt)
