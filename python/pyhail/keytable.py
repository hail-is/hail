from pyhail.java import scala_object

class KeyTable:
    """:class:`.KeyTable` is Hail's version of a SQL
    table where fields can be designated as keys. 
    
    :param :class:`.HailContext` hc: Hail spark context.
    :param JavaKeyTable jkt: Java KeyTable object.
    """

    def __init__(self, hc, jkt):
        self.hc = hc
        self.jkt = jkt

    def nFields(self):
        """Number of fields in the key-table

        :return: int
        """
        return self.jkt.nFields()

    def schema(self):
        """Key-table schema

        :return: ???
        """
        return self.jkt.schema()

    def keyNames(self):
        """Field names that are keys

        :return: list[str]
        """
        return self.jkt.keyNames()

    def fieldNames(self):
        """Field names

        :return: list[str]
        """
        return self.jkt.fieldNames()

    def nRows(self):
        """Number of rows in the key-table

        :return: long
        """
        return self.jkt.nRows()

    def same(self, other):
        """Test whether two key-tables are identical

        :param :class:`.KeyTable` other: KeyTable to compare to

        :return: bool
        """
        return self.jkt.same(other.jkt)

    def export(self, output, types_file = None):
        """Export key-table to a TSV file.

        :param str output: Output file path

        :param str types_file: Output path of types file

        :rtype: Nothing.
        """
        self.jkt.export(self.hc.jsc, output, types_file)

    def filter(self, code, keep = True):
        """Filter rows from key-table.

        :param str code: Annotation expression.

        :param bool keep: Keep rows where annotation expression evaluates to True

        :return: :class:`.KeyTable`
        """
        return KeyTable(self.hc, self.jkt.filter(code, keep))

    def annotate(self, code, key_names = None):
        """Add fields to key-table.

        :param str code: Annotation expression.

        :param str key_names: Comma separated list of field names to be treated as a key

        :return: :class:`.KeyTable`
        """
        return KeyTable(self.hc, self.jkt.annotate(code, key_names))

    def join(self, right, how = 'inner'):
        """Join two key-tables together. Both key-tables must have identical key schemas
        and non-overlapping fields in order to be joined.

        :param :class:`.KeyTable` right: Key-table to join

        :param str how: Method for joining two tables together. One of "inner", "outer", "left", "right".

        :return: :class:`.KeyTable`
        """
        return KeyTable(self.hc, self.jkt.join(right.jkt, how))

    def aggregate(self, key_cond, agg_cond):
        """Group by key condition and aggregate results 

        :param str key_cond: Named expression defining keys in the new key-table

        :param str agg_cond: Named expression specifying how new fields are computed

        :return: :class:`.KeyTable`
        """
        return KeyTable(self.hc, self.jkt.aggregate(key_cond, agg_cond))

    def forall(self, code):
        """Tests whether a condition is true for all rows

        :param str code: Boolean expression

        :return: bool
        """
        return self.jkt.forall(cond)

    def exists(self, code):
        """Tests whether a condition is true for any row

        :param str code: Boolean expression

        :return: bool
        """
        return self.jkt.exists(cond)