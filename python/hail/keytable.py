from __future__ import print_function  # Python 2 and 3 print compatibility

from py4j.protocol import Py4JJavaError
from pyspark.sql import DataFrame
from hail.java import scala_package_object, raise_py4j_exception
from hail.type import Type


class KeyTable(object):
    """Hail's version of a SQL table where columns can be designated as keys.

    :param hc: Hail context
    :type hc: :class:`.HailContext`
    :param jkt: Java key table
    """

    def __init__(self, hc, jkt):
        self.hc = hc
        self.jkt = jkt

    def __repr__(self):
        try:
            return self.jkt.toString()
        except Py4JJavaError as e:
            raise_py4j_exception(e)

    def num_columns(self):
        """Number of columns.

        :rtype: int
        """
        try:
            return self.jkt.nFields()
        except Py4JJavaError as e:
            raise_py4j_exception(e)

    def schema(self):
        """KeyTable schema.

        >>> print(kt.schema())

        :rtype: :class:`.Type`
        """
        try:
            return Type._from_java(self.jkt.signature())
        except Py4JJavaError as e:
            raise_py4j_exception(e)

    def key_names(self):
        """Column names that are keys.

        :rtype: list of str

        """
        try:
            return list(self.jkt.keyNames())
        except Py4JJavaError as e:
            raise_py4j_exception(e)

    def column_names(self):
        """Names of all columns.

        :rtype: list of str
        """
        try:
            return list(self.jkt.fieldNames())
        except Py4JJavaError as e:
            raise_py4j_exception(e)

    def num_rows(self):
        """Number of rows.

        :rtype: long
        """
        try:
            return self.jkt.nRows()
        except Py4JJavaError as e:
            raise_py4j_exception(e)

    def same(self, other):
        """Test whether two KeyTables are identical.

        **Examples**

        >>> kt1 = hc.import_keytable("data/example1.tsv")
        >>> kt2 = hc.import_keytable("data/example2.tsv")
        >>> if kt1.same(kt2):
        >>>     print_function("KeyTables are the same!")

        :param other: key table to compare against
        :type other: :class:`.KeyTable` 

        :rtype: bool
        """
        try:
            return self.jkt.same(other.jkt)
        except Py4JJavaError as e:
            raise_py4j_exception(e)

    def export(self, output, types_file=None):
        """Export to a TSV file.

        **Examples**

        Rename column names of KeyTable and export to file:

        >>> (hc.import_keytable("data/example.tsv")
        >>>    .rename({'column1' : 'newColumn1'})
        >>>    .export("data/kt1_renamed.tsv"))

        :param str output: Output file path.
        :param str types_file: Output path of types file.
        """
        try:
            self.jkt.export(self.hc._jsc, output, types_file)
        except Py4JJavaError as e:
            raise_py4j_exception(e)

    def filter(self, condition, keep=True):
        """Filter rows.

        **Examples**

        Keep rows where ``C1`` equals 5:

        >>> kt = (hc.import_keytable("data/example.tsv")
        >>>         .filter("C1 == 5"))

        Remove rows where ``C1`` equals 10:

        >>> kt = (hc.import_keytable("data/example.tsv")
        >>>         .filter("C1 == 10", keep=False))

        **Notes**

        The scope for ``condition`` is all column names in the input :class:`KeyTable`.

        For more information, see the documentation on writing `expressions <../overview.html#expressions>`_
        and using the `Hail Expression Language <../reference.html#HailExpressionLanguage>`_.

        .. caution::
           When ``condition`` evaluates to missing, the row will be removed regardless of whether ``keep=True`` or ``keep=False``.

        :param str condition: Annotation expression.
        :param bool keep: Keep rows where ``condition`` evaluates to True.

        :return: A key table whose rows have been filtered by evaluating ``condition``.
        :rtype: :class:`.KeyTable`
        """
        try:
            return KeyTable(self.hc, self.jkt.filter(condition, keep))
        except Py4JJavaError as e:
            raise_py4j_exception(e)

    def annotate(self, condition):
        """Add new columns computed from existing columns.

        **Examples**

        Add new column ``Y`` which is equal to 5 times ``X``:

        >>> kt = (hc.import_keytable("data/example.tsv")
        >>>         .annotate("Y = 5 * X"))

        **Notes**

        The scope for ``condition`` is all column names in the input :class:`KeyTable`.

        For more information, see the documentation on writing `expressions <../overview.html#expressions>`_
        and using the `Hail Expression Language <../reference.html#HailExpressionLanguage>`_.

        :param condition: Annotation expression or multiple annotation expressions.
        :type condition: str or list of str

        :return: A key table with new columns specified by ``condition``.
        :rtype: :class:`.KeyTable`
        """
        if isinstance(condition, list):
            condition = ','.join(condition)

        try:
            return KeyTable(self.hc, self.jkt.annotate(condition))
        except Py4JJavaError as e:
            raise_py4j_exception(e)

    def join(self, right, how='inner'):
        """Join two KeyTables together.

        **Examples**

        Join ``kt1`` to ``kt2`` to produce ``kt3``:

        >>> kt1 = hc.import_keytable("data/example1.tsv")
        >>> kt2 = hc.import_keytable("data/example2.tsv")
        >>> kt3 = kt1.join(kt2)

        **Notes:**

        Hail supports four types of joins specified by ``how``:

         - **inner** -- Key must be present in both ``kt1`` and ``kt2``.
         - **outer** -- Key present in ``kt1`` or ``kt2``. For keys only in ``kt1``, the value of non-key columns from ``kt2`` is set to missing.
           Likewise, for keys only in ``kt2``, the value of non-key columns from ``kt1`` is set to missing.
         - **left** -- Key present in ``kt1``. For keys only in ``kt1``, the value of non-key columns from ``kt2`` is set to missing.
         - **right** -- Key present in ``kt2``. For keys only in ``kt2``, the value of non-key columns from ``kt1`` is set to missing.

        .. note::
            Both KeyTables must have identical key schemas and non-overlapping column names.

        :param  right: KeyTable to join
        :type right: :class:`.KeyTable`
        :param str how: Method for joining two tables together. One of "inner", "outer", "left", "right".

        :return: A key table that is the result of joining this key table with another.
        :rtype: :class:`.KeyTable`
        """
        try:
            return KeyTable(self.hc, self.jkt.join(right.jkt, how))
        except Py4JJavaError as e:
            raise_py4j_exception(e)

    def aggregate_by_key(self, key_condition, agg_condition):
        """Group by key condition and aggregate results.

        **Examples**

        Compute mean height by sex:

        >>> kt = hc.import_keytable("data/example.tsv")
        >>> kt_ht_by_sex = kt.aggregate_by_key("SEX = SEX", "MEAN_HT = HT.stats().mean")

        The KeyTable ``kt`` has the following data:

        +--------+----------+----------+
        |   ID   |    HT    |    SEX   |
        +========+==========+==========+
        |   1    |    65    |     M    |
        +--------+----------+----------+
        |   2    |    72    |     M    |
        +--------+----------+----------+
        |   3    |    70    |     F    |
        +--------+----------+----------+
        |   4    |    60    |     F    |
        +--------+----------+----------+

        The result of :py:meth:`.aggregate_by_key` is a KeyTable ``kt_ht_by_sex`` with the following data:

        +--------+----------+
        |   SEX  |MEAN_HT   |
        +========+==========+
        |   M    |  68.5    |
        +--------+----------+
        |   F    |   65     |
        +--------+----------+

        **Notes**

        The scope for both ``key_condition`` and ``agg_condition`` is all column names in the input :class:`KeyTable`.

        For more information, see the documentation on writing `expressions <../overview.html#expressions>`_
        and using the `Hail Expression Language <../reference.html#HailExpressionLanguage>`_.

        :param key_condition: Named expression(s) for how to compute the keys of the new KeyTable.
        :type key_condition: str or list of str
        :param agg_condition: Named aggregation expression(s).
        :type agg_condition: str or list of str

        :return: A new KeyTable with the keys computed from the ``key_condition`` and the remaining columns computed from the ``agg_condition``.
        :rtype: :class:`.KeyTable`
        """
        if isinstance(key_condition, list):
            key_condition = ",".join(key_condition)

        if isinstance(agg_condition, list):
            agg_condition = ", ".join(agg_condition)

        try:
            return KeyTable(self.hc, self.jkt.aggregate(key_condition, agg_condition))
        except Py4JJavaError as e:
            raise_py4j_exception(e)

    def forall(self, code):
        """Test whether a condition is true for all rows.

        **Examples**

        Test whether all rows in the KeyTable have the value of ``C1`` equal to 5:

        >>> kt = hc.import_keytable('data/example.tsv')
        >>> if kt.forall("C1 == 5"):
        >>>     print_function("All rows have C1 equal 5.")

        :param str code: Boolean expression.

        :rtype: bool
        """
        try:
            return self.jkt.forall(code)
        except Py4JJavaError as e:
            raise_py4j_exception(e)

    def exists(self, code):
        """Test whether a condition is true for any row.

        **Examples**

        Test whether any row in the KeyTable has the value of ``C1`` equal to 5:

        >>> kt = hc.import_keytable('data/example.tsv')
        >>> if kt.exists("C1 == 5"):
        >>>     print_function("At least one row has C1 equal 5.")

        :param str code: Boolean expression.

        :rtype: bool
        """
        try:
            return self.jkt.exists(code)
        except Py4JJavaError as e:
            raise_py4j_exception(e)

    def rename(self, column_names):
        """Rename columns of KeyTable.

        ``column_names`` can be either a list of new names or a dict
        mapping old names to new names.  If ``column_names`` is a list,
        its length must be the number of columns in this ``KeyTable``.

        **Examples**

        Rename using a list:

        >>> kt = hc.import_keytable('data/example.tsv')
        >>> kt_renamed = kt.rename(['newColumn1', 'newColumn2', 'newColumn3'])

        Rename using a dict:

        >>> kt = hc.import_keytable('data/example.tsv')
        >>> kt_renamed = kt.rename({'column1' : 'newColumn1'})

        :param column_names: list of new column names or a dict mapping old names to new names.
        :type list of str or dict of str: str

        :return: A key table with renamed columns.
        :rtype: :class:`.KeyTable`
        """
        try:
            return KeyTable(self.hc, self.jkt.rename(column_names))
        except Py4JJavaError as e:
            raise_py4j_exception(e)

    def expand_types(self):
        """Expand types Locus, Interval, AltAllele, Variant, Genotype, Char,
        Set and Dict.  Char is converted to String.  Set is converted
        to Array.  Dict[T] is converted to

        .. code-block:: text

            Array[Struct {
                key: String
                value: T
            }]

        :return: key table with signature containing only types:
          Boolean, Int, Long, Float, Double, Array and Struct
        :rtype: :class:`.KeyTable`
        """
        try:
            return KeyTable(self.hc, self.jkt.expandTypes())
        except Py4JJavaError as e:
            raise_py4j_exception(e)

    def key_by(self, key_names):
        """Change which columns are keys.

        **Examples**

        Assume ``kt`` is a ``KeyTable`` with three columns: c1, c2 and
        c3 and key c1.

        Change key columns:

        >>> kt.key_by(['c2', 'c3'])

        Set to no keys:

        >>> kt.key_by([])

        **Notes**

        The order of the columns will be the original order with the key
        columns moved to the beginning in the order given by ``key_names``.

        :param key_names: List of columns to be used as keys.
        :type key_names: list of str

        :return: A key table whose key columns are given by ``key_names``.
        :rtype: :class:`.KeyTable`

        """
        try:
            return KeyTable(self.hc, self.jkt.select(self.column_names(), key_names))
        except Py4JJavaError as e:
            raise_py4j_exception(e)

    def flatten(self):
        """Flatten nested Structs.  Column names will be concatenated with dot
        (.).

        **Example**

        Flatten Structs in KeyTable:

        >>> (hc.import_keytable("data/example.tsv")
        >>>    .flatten())

        Consider a KeyTable ``kt`` with signature

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

        :return: A key table with no columns of type Struct.
        :rtype: :class:`.KeyTable`

        """
        try:
            return KeyTable(self.hc, self.jkt.flatten())
        except Py4JJavaError as e:
            raise_py4j_exception(e)

    def select(self, column_names):
        """Select a subset of columns.

        **Examples**

        Assume ``kt`` is a ``KeyTable`` with three columns: C1, C2 and
        C3.

        Select/drop columns:

        >>> new_kt = kt.select(['C1'])

        Reorder the columns:

        >>> new_kt = kt.select(['C3', 'C1', 'C2'])

        Drop all columns:

        >>> new_kt = kt.select([])

        **Notes**

        The order of the columns will be the order given
        by ``column_names`` with the key columns moved to the beginning
        in the order of the key columns in this ``KeyTable``.

        :param column_names: List of columns to be selected.
        :type: list of str

        :return: A key table with selected columns in the order given by ``column_names``.
        :rtype: :class:`.KeyTable`

        """
        try:
            new_key_names = [k for k in self.key_names() if k in column_names]
            return KeyTable(self.hc, self.jkt.select(column_names, new_key_names))
        except Py4JJavaError as e:
            raise_py4j_exception(e)

    def to_dataframe(self, expand=True, flatten=True):
        """Converts this KeyTable to a Spark DataFrame.

        :param bool expand: If true, expand_types before converting to
          DataFrame.
        :param bool flatten: If true, flatten before converting to
          DataFrame.  If both are true, flatten is run after expand so
          that expanded types are flattened.

        :rtype: :class:`pyspark.sql.DataFrame`

        """
        try:
            jkt = self.jkt
            if expand:
                jkt = jkt.expandTypes()
            if flatten:
                jkt = jkt.flatten()
            return DataFrame(jkt.toDF(self.hc._jsql_context), self.hc._sql_context)
        except Py4JJavaError as e:
            raise_py4j_exception(e)

    def to_pandas(self, expand = True, flatten = True):
        """Converts this KeyTable into a Pandas DataFrame.

        :param bool expand: If true, expand_types before converting to
          Pandas DataFrame.

        :param bool flatten: If true, flatten before converting to Pandas
          DataFrame.  If both are true, flatten is run after expand so
          that expanded types are flattened.

        :return: pandas.DataFrame
        """
        return self.to_dataframe(expand, flatten).toPandas()

    def export_mongodb(self, mode='append'):
        """Export to MongoDB"""
        (scala_package_object(self.hc._hail.driver)
         .exportMongoDB(self.hc._jsql_context, self.jkt, mode))

    def explode(self, column_names):
        """Explode columns of this KeyTable.

        The explode operation unpacks the elements in a column of type ``Array`` or ``Set`` into its own row.
        If an empty ``Array`` or ``Set`` is exploded, the entire row is removed from the :py:class:`.KeyTable`.

        **Examples**

        Assume ``kt`` is a :py:class:`.KeyTable` with three columns: c1, c2 and
        c3. The types of each column are ``String``, ``Array[Int]``, and ``Array[Array[Int]]`` respectively.
        c1 cannot be exploded because its type is not an ``Array`` or ``Set``.
        c2 can only be exploded once because the type of c2 after the first explode operation is ``Int``.

        +----+----------+----------------+
        | c1 |   c2     |   c3           |
        +====+==========+================+
        |  a | [1,2,NA] |[[3,4], []]     |
        +----+----------+----------------+

        Explode c2:

        >>> exploded_kt = (hc.import_keytable("data/example.tsv")
        >>>                  .explode('c2'))

        +----+-------+-----------------+
        | c1 |   c2  |    c3           |
        +====+=======+=================+
        |  a | 1     | [[3,4], []]     |
        +----+-------+-----------------+
        |  a | 2     | [[3,4], []]     |
        +----+-------+-----------------+

        Explode c2 once and c3 twice:

        >>> exploded_kt = (hc.import_keytable("data/example.tsv")
        >>>                  .explode(['c2', 'c3', 'c3']))

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
            
        :return: A key table with columns exploded.
        :rtype: :class:`.KeyTable`
        """

        try:
            if isinstance(column_names, str):
                column_names = [column_names]
            return KeyTable(self.hc, self.jkt.explode(column_names))
        except Py4JJavaError as e:
            raise_py4j_exception(e)
