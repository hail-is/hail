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
            return self.jkt.keyNames()
        except Py4JJavaError as e:
            self._raise_py4j_exception(e)

    def field_names(self):
        """Names of all fields in the key-table

        :rtype: list of str
        """
        try:
            return self.jkt.fieldNames()
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

    def annotate(self, code, key_names=''):
        """Add fields to key-table.

        :param str code: Annotation expression.

        :param str key_names: Comma separated list of field names to be treated as a key

        :rtype: :class:`.KeyTable`
        """
        try:
            return KeyTable(self.hc, self.jkt.annotate(code, key_names))
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
            key_code = ", ".join([str(l) for l in list])

        if isinstance(agg_code, list):
            agg_code = ", ".join([str(l) for l in list])

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
        """Rename fields of KeyTable.

        Specify update mapping with a list explicitly specifying all field names.
        This list must be the same length as the number of fields in the KeyTable.

        >>> kt = hc.import_keytable("data/kt1.tsv")
        >>> new_field_names = kt.field_names
        >>> new_field_names[0] = "newName"
        >>> kt_renamed = kt.rename(new_field_names)

        Specify update mapping with a dict corresponding to old_name : new_name.
        Only field names present in the update mapping will be altered.

        >>> kt = hc.import_keytable("data/kt1.tsv")
        >>> kt_renamed = kt.rename({"field1" : "newName"})

        :param field_names: Mapping of original field name
        :type list of str or dict of str:str

        :return: A KeyTable with updated field names.

        :rtype: KeyTable
        """
        try:
            return KeyTable(self.hc, self.jkt.rename(field_names))
        except Py4JJavaError as e:
            self._raise_py4j_exception(e)

    def select(self, field_names, key_names):
        """Reorder and drop fields of KeyTable.

        Select which fields to keep and the order they appear in. Also, choose which fields are considered keys.

        In the examples below, assume **kt** is a KeyTable with three fields ["field1", "field2", "field3"] where ["field1", "field2"] are the key fields.

        **Example 1: Change which field is considered the key**
        >>> new_kt = kt.select(["field1", "field2", "field3"], ["field3"])

        **Example 2: Change the order of the fields. Note, key fields always come before non-key fields.**
        >>> new_kt = kt.select(["field3", "field1", "field2"], ["field1", "field2"])

        **Example 3: Drop fields from the KeyTable**
        >>> new_kt = kt.select(["field1"], ["field1"])

        **Example 4: Do not have fields designated as key fields**
        >>> new_kt = kt.select(["field1", "field2", "field3"], [])

        **Example 5: Change the order of the key fields**
        >>> new_kt = kt.select(["field1", "field2", "field3"], ["field2", "field1"])

        :param field_names: Names of fields to be included in new KeyTable. Order of fields is preserved except key fields always come before non-key fields.
        :type list of str

        :param key_names: Names of fields to be designated as keys. Order of fields is preserved.
        :type list of str

        :return: A KeyTable with only selected fields in specified order.

        :rtype: KeyTable
        """
        try:
            return KeyTable(self.hc, self.jkt.select(field_names, key_names))
        except Py4JJavaError as e:
            self._raise_py4j_exception(e)

