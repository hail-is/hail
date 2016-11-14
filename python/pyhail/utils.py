
class Type(object):
    def __init__(self, jtype):
        self.jtype = jtype

    def __repr__(self):
        return self.jtype.toString()

    def __str__(self):
        return self.jtype.toPrettyString(False, False)

class TextTableConfig(object):
    """:class:`.TextTableConfig` specifies additional options for importing TSV files.

    :param bool noheader: File has no header and columns should be indicated by `_1, _2, ... _N' (0-indexed)

    :param bool impute: Impute column types from the file

    :param str comment: Skip lines beginning with the given pattern

    :param str delimiter: Field delimiter regex

    :param str missing: Specify identifier to be treated as missing

    :param str types: Define types of fields in annotations files   
    """
    def __init__(self, noheader = False, impute = False,
                 comment = None, delimiter = "\t", missing = "NA", types = None):
        self.noheader = noheader
        self.impute = impute
        self.comment = comment
        self.delimiter = delimiter
        self.missing = missing
        self.types = types

    def __str__(self):
        res = ["--comment", self.comment, "--delimiter", self.delimiter,
               "--missing", self.missing]

        if self.noheader:
            res.append("--no-header")

        if self.impute:
            res.append("--impute")

        return " ".join(res)

    def _toJavaObject(self, hc):
        """Convert to java TextTableConfiguration object

        :param :class:`.HailContext` hc: Hail spark context.
        """
        return hc.jvm.org.broadinstitute.hail.utils.TextTableConfiguration.apply(self.types, self.comment,
                                                             self.delimiter, self.missing,
                                                             self.noheader, self.impute)

