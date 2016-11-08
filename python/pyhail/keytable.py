from pyhail.java import scala_object

class KeyTable:
    """:class:`.KeyTable` ... 
    
    :param SparkContext sc: The pyspark context.
    :param JavaKeyTable jkt: The java key table object.
    """

    def __init__(self, hc, jkt):
        self.hc = hc
        self.jkt = jkt

    # FIXME schema stuff...
    def nKeys(self):
        return self.jkt.nKeys()

    def nValues(self):
        return self.jkt.nValues()

    def nFields(self):
        return self.jkt.nFields()

    def schema(self):
        return self.jkt.schema()

    def keyNames(self):
        return self.jkt.keyNames()

    def valueNames(self):
        return self.jkt.valueNames()

    def nRows(self):
        """Number of rows in the key-table

        :return: long
        """
        return self.jkt.nRows()

    def same(self, other):
        """Compares two key-tables

        :param KeyTable other: KeyTable to compare to

        :return: bool
        """
        return self.jkt.same(other.jkt)

    def export(self, output, types_file = None):
        """Export key-table to a tsv file.

        :param str output: Output file path

        :param str types_file: Output path of types file

        :return: Nothing.
        """
        self.jkt.export(self.hc.jsc, output, types_file)

    def filter(self, code, keep = True):
        """Filter rows from key-table.

        :param str code: Annotation expression.

        :param bool keep: Keep rows where annotation expression evaluates to True

        :return: KeyTable
        """
        return KeyTable(self.hc, self.jkt.filter(code, keep))

    def annotate(self, code, key_names = None):
        """Add fields to key-table.

        :param str code: Annotation expression.

        :param bool keep: Keep rows where annotation expression evaluates to True

        :return: KeyTable
        """
        return KeyTable(self.hc, self.jkt.annotate(code, key_names))

    def join(self, right, how = 'inner'):
        """Join two key-tables together. Both key-tables must have identical key schemas
        and non-overlapping fields in order to be joined.

        :param KeyTable right: key-table to join

        :param str how: Method for joining two tables together. One of "inner", "outer", "left", "right".

        :return: KeyTable
        """
        ## Check keys are same

        ## Check fields do not overlap

        if how == "inner":
            return KeyTable(self.hc, self.jkt.innerJoin(right.jkt))
        elif how == "outer":
            return KeyTable(self.hc, self.jkt.outerJoin(right.jkt))
        elif how == "left":
            return KeyTable(self.hc, self.jkt.leftJoin(right.jkt))
        elif how == "right":
            return KeyTable(self.hc, self.jkt.rightJoin(right.jkt))
        else:
            pass



#    def import_fam(hc, path, ...):
#        pass


    # kt.select(star().except('a'), expr('sum', 'x + b'), expr('a', 'a.b.c = 9'))
    # kt.select(star().except('a'), {'sum': 'x + b', 'a': 'update(a.b.c, 9)'})


    # def for_all(self, condition):
    #     pass

    # FIXME returns TypedValue
    # def aggregate(value expressions...):
    #     pass
    #
    # def aggregate_by_key(self, value expressions...):
    #     pass
