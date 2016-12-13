from py4j.protocol import Py4JJavaError
from pyspark.sql import DataFrame
from pyhail.java import scala_package_object
from pyhail.type import Type

class KeyTable(object):
    """:class:`.KeyTable` is Hail's version of a SQL table where fields
    can be designated as keys.

    """

    def __init__(self, hc, jkt):
        """
        :param HailContext hc: Hail spark context.

        :param JavaKeyTable jkt: Java KeyTable object.
        """
        self.hc = hc
        self.jkt = jkt

    def _raise_py4j_exception(self, e):
        self.hc._raise_py4j_exception(e)

    def __repr__(self):
        try:
            return self.jkt.toString()
        except Py4JJavaError as e:
            self._raise_py4j_exception(e)

    def nfields(self):
        """Number of fields in the key-table

        :rtype: int
        """
        try:
            return self.jkt.nFields()
        except Py4JJavaError as e:
            self._raise_py4j_exception(e)

    def schema(self):
        """Key-table schema

        :rtype: :class:`.Type`
        """
        try:
            return Type(self.jkt.signature())
        except Py4JJavaError as e:
            self._raise_py4j_exception(e)

    def key_names(self):
        """Field names that are keys

        :rtype: list of str
        """
        try:
            return list(self.jkt.keyNames())
        except Py4JJavaError as e:
            self._raise_py4j_exception(e)

    def field_names(self):
        """Names of all fields in the key-table

        :rtype: list of str
        """
        try:
            return list(self.jkt.fieldNames())
        except Py4JJavaError as e:
            self._raise_py4j_exception(e)

    def nrows(self):
        """Number of rows in the key-table

        :rtype: long
        """
        try:
            return self.jkt.nRows()
        except Py4JJavaError as e:
            self._raise_py4j_exception(e)
        

    def same(self, other):
        """Test whether two key-tables are identical

        :param other: KeyTable to compare to
        :type other: :class:`.KeyTable` 

        :rtype: bool
        """
        try:
            return self.jkt.same(other.jkt)
        except Py4JJavaError as e:
            self._raise_py4j_exception(e)

    def export(self, output, types_file=None):
        """Export key-table to a TSV file.

        :param str output: Output file path

        :param str types_file: Output path of types file

        :rtype: Nothing.
        """
        try:
            self.jkt.export(self.hc.jsc, output, types_file)
        except Py4JJavaError as e:
            self._raise_py4j_exception(e)

    def filter(self, code, keep=True):
        """Filter rows from key-table.

        :param str code: Annotation expression.

        :param bool keep: Keep rows where annotation expression evaluates to True

        :rtype: :class:`.KeyTable`
        """
        try:
            return KeyTable(self.hc, self.jkt.filter(code, keep))
        except Py4JJavaError as e:
            self._raise_py4j_exception(e)

    def annotate(self, code):
        """Add fields to key-table.

        :param str code: Annotation expression.

        :rtype: :class:`.KeyTable`
        """
        try:
            return KeyTable(self.hc, self.jkt.annotate(code))
        except Py4JJavaError as e:
            self._raise_py4j_exception(e)

    def join(self, right, how='inner'):
        """Join two key-tables together. Both key-tables must have identical key schemas
        and non-overlapping field names.

        :param  right: Key-table to join
        :type right: :class:`.KeyTable`

        :param str how: Method for joining two tables together. One of "inner", "outer", "left", "right".

        :rtype: :class:`.KeyTable`
        """
        try:
            return KeyTable(self.hc, self.jkt.join(right.jkt, how))
        except Py4JJavaError as e:
            self._raise_py4j_exception(e)

    def aggregate_by_key(self, key_code, agg_code):
        """Group by key condition and aggregate results 

        :param key_code: Named expression(s) for which fields are keys.
        :type key_code: str or list of str

        :param agg_code: Named aggregation expression(s).
        :type agg_code: str or list of str

        :rtype: :class:`.KeyTable`
        """
        if isinstance(key_code, list):
            key_code = ",".join(key_code)

        if isinstance(agg_code, list):
            agg_code = ", ".join(agg_code)

        try:
            return KeyTable(self.hc, self.jkt.aggregate(key_code, agg_code))
        except Py4JJavaError as e:
            self._raise_py4j_exception(e)

    def forall(self, code):
        """Tests whether a condition is true for all rows

        :param str code: Boolean expression

        :rtype: bool
        """
        try:
            return self.jkt.forall(code)
        except Py4JJavaError as e:
            self._raise_py4j_exception(e)

    def exists(self, code):
        """Tests whether a condition is true for any row

        :param str code: Boolean expression

        :rtype: bool
        """
        try:
            return self.jkt.exists(code)
        except Py4JJavaError as e:
            self._raise_py4j_exception(e)

    def rename(self, field_names):
        """Rename fields of ``KeyTable``.

        ``field_names`` can be either a list of new names or a dict
        mapping old names to new names.  If ``field_names`` is a list,
        its length must be the number of fields in this ``KeyTable``.

        **Examples**

        Rename using a list:

        >>> kt = hc.import_keytable('data/kt1.tsv')
        >>> kt_renamed = kt.rename(['newField1', 'newField2', 'newField3'])

        Rename using a dict:

        >>> kt = hc.import_keytable('data/kt1.tsv')
        >>> kt_renamed = kt.rename({'field1' : 'newField1'})

        :param field_names: list of new field names or a dict mapping old names to new names.
        :type list of str or dict of str: str

        :return: A KeyTable with renamed fields.

        :rtype: KeyTable
        """
        try:
            return KeyTable(self.hc, self.jkt.rename(field_names))
        except Py4JJavaError as e:
            self._raise_py4j_exception(e)

    def expand_types(self):
        """Expand types Locus, Interval, AltAllele, Variant, Genotype, Char,
        Set and Dict.  Char is converted to String.  Set is converted
        to Array.  Dict[T] is converted to::

          Array[Struct {
            key: String
            value: T
          }]

        :return: KeyTable with signature containing only types:
          Boolean, Int, Long, Float, Double, Array and Struct

        :rtype: KeyTable
        """
        try:
            return KeyTable(self.hc, self.jkt.expandTypes())
        except Py4JJavaError as e:
            self._raise_py4j_exception(e)

    def key_by(self, key_names):
        """Return a new ``KeyTable`` with keys given by ``key_names``.  The
        order of the fields will be the original order with the key
        fields moved to the beginning in the order given by
        ``key_names``.

        **Examples**

        Assume ``kt`` is a ``KeyTable`` with three fields: f1, f2 and
        f3 and key f1.

        Change key fields:

        >>> kt.key_by(['f2', 'f3'])

        Set to no keys:

        >>> kt.key_by([])

        :param key_names: List of fields to be used as keys.
        :type key_names: list of str

        :return: A ``KeyTable`` whose key fields are given by
          ``key_names``.

        :rtype: KeyTable

        """
        try:
            return KeyTable(self.hc, self.jkt.select(self.field_names(), key_names))
        except Py4JJavaError as e:
            self._raise_py4j_exception(e)

    def flatten(self):
        """Flatten nested Structs.  Field names will be concatenated with dot
        (.).

        **Example**

        Consider a KeyTable with signature::

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

        and a single key column ``a``.  The result of flatten is be::

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

        :return: A KeyTable with no columns of type Struct.

        :rtype: KeyTable

        """
        try:
            return KeyTable(self.hc, self.jkt.flatten())
        except Py4JJavaError as e:
            self._raise_py4j_exception(e)

    def select(self, field_names):
        """Selects a subset of fields of this ``KeyTable`` and returns a new
        ``KeyTable``.  The order of the fields will be the order given
        by ``field_names`` with the key fields moved to the beginning
        in the order of the key fields in this ``KeyTable``.

        **Examples**

        Assume ``kt`` is a ``KeyTable`` with three fields: f1, f2 and
        f3.

        Select/drop fields:

        >>> new_kt = kt.select(['f1'])

        Reorder the fields:

        >>> new_kt = kt.select(['f3', 'f1', 'f2'])

        Drop all fields:

        >>> new_kt = kt.select([])

        :param field_names: List of fields to be selected.
        :type: list of str

        :return: A ``KeyTable`` with selected fields in the order
          given by ``field_names``.

        :rtype: KeyTable

        """
        new_key_names = [k for k in self.key_names() if k in field_names]
        
        try:
            return KeyTable(self.hc, self.jkt.select(field_names, new_key_names))
        except Py4JJavaError as e:
            self._raise_py4j_exception(e)

    def to_dataframe(self, expand=True, flatten=True):
        """Converts this KeyTable to a Spark DataFrame.

        :param bool expand: If true, expand_types before converting to
          DataFrame.

        :param bool flatten: If true, flatten before converting to
          DataFrame.  If both are true, flatten is run after expand so
          that expanded types are flattened.

        :rtype: DataFrame

        """
        try:
            jkt = self.jkt
            if expand:
                jkt = jkt.expandTypes()
            if flatten:
                jkt = jkt.flatten()
            return DataFrame(jkt.toDF(self.hc.jsql_context), self.hc.sql_context)
        except Py4JJavaError as e:
            self._raise_py4j_exception(e)

    def export_mongodb(self, mode='append'):
        (scala_package_object(self.hc.jvm.org.broadinstitute.hail.driver)
         .exportMongoDB(self.hc.jsql_context, self.jkt, mode))

    def explode(self, column_names):
        """Explodes columns of this ``KeyTable``

        The explode operation unpacks the elements in a column of type ``Array`` or ``Set`` into its own row.
        If an empty ``Array`` or ``Set`` is exploded, the entire row is removed from the ``KeyTable``.

        **Examples**

        Assume ``kt`` is a ``KeyTable`` with three columns: c1, c2 and
        c3. The types of each column are ``String``, ``Array[Int]``, and ``Array[Array[Int]]`` respectively.
        c1 cannot be exploded because its type is not an ``Array`` or ``Set``.
        c2 can only be exploded once because the type of c2 after the first explode operation is ``Int``.

        +----+----------+----------------+
        | c1 |   c2     |   c3           |
        +====+==========+================+
        |  a | [1,2,NA] |[[3,4], []]     |
        +----+----------+----------------+

        Explode c2 once and c3 twice:

        >>> new_kt = kt.explode(['c2', 'c3', 'c3'])

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

        :param: column_names: List of columns to be exploded.
        :type: list of str

        :return: A KeyTable with columns exploded.

        :rtype: KeyTable

        """

        try:
            return KeyTable(self.hc, self.jkt.explode(column_names))
        except Py4JJavaError as e:
            self._raise_py4j_exception(e)

