from pyhail.utils import Type

class KeyTable(object):
    """:class:`.KeyTable` is Hail's version of a SQL
    table where fields can be designated as keys.
    """

    def __init__(self, hc, jkt):
        """
        :param hc: Hail spark context.
        :type hc: :class:`.HailContext`

        :param JavaKeyTable jkt: Java KeyTable object.
        """
        self.hc = hc
        self.jkt = jkt

    def __repr__(self):
        return self.jkt.toString()

    def nfields(self):
        """Number of fields in the key-table

        :rtype: int
        """
        return self.jkt.nFields()

    def schema(self):
        """Key-table schema

        :rtype: :class:`.Type`
        """
        return Type(self.jkt.signature())

    def key_names(self):
        """Field names that are keys

        :rtype: list of str
        """
        return self.jkt.keyNames()

    def field_names(self):
        """Names of all fields in the key-table

        :rtype: list of str
        """
        return self.jkt.fieldNames()

    def nrows(self):
        """Number of rows in the key-table

        :rtype: long
        """
        return self.jkt.nRows()

    def same(self, other):
        """Test whether two key-tables are identical

        :param other: KeyTable to compare to
        :type other: :class:`.KeyTable` 

        :rtype: bool
        """
        return self.jkt.same(other.jkt)

    def export(self, output, types_file=None):
        """Export key-table to a TSV file.

        :param str output: Output file path

        :param str types_file: Output path of types file

        :rtype: Nothing.
        """
        self.jkt.export(self.hc.jsc, output, types_file)

    def filter(self, code, keep=True):
        """Filter rows from key-table.

        :param str code: Annotation expression.

        :param bool keep: Keep rows where annotation expression evaluates to True

        :rtype: :class:`.KeyTable`
        """
        return KeyTable(self.hc, self.jkt.filter(code, keep))

    def annotate(self, code, key_names=None):
        """Add fields to key-table.

        :param str code: Annotation expression.

        :param str key_names: Comma separated list of field names to be treated as a key

        :rtype: :class:`.KeyTable`
        """
        return KeyTable(self.hc, self.jkt.annotate(code, key_names))

    def join(self, right, how='inner'):
        """Join two key-tables together. Both key-tables must have identical key schemas
        and non-overlapping field names.

        :param  right: Key-table to join
        :type right: :class:`.KeyTable`

        :param str how: Method for joining two tables together. One of "inner", "outer", "left", "right".

        :rtype: :class:`.KeyTable`
        """
        return KeyTable(self.hc, self.jkt.join(right.jkt, how))

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

        return KeyTable(self.hc, self.jkt.aggregate(key_code, agg_code))

    def forall(self, code):
        """Tests whether a condition is true for all rows

        :param str code: Boolean expression

        :rtype: bool
        """
        return self.jkt.forall(code)

    def exists(self, code):
        """Tests whether a condition is true for any row

        :param str code: Boolean expression

        :rtype: bool
        """
        return self.jkt.exists(code)