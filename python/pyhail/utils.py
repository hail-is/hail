
class TextTableConfig(object):
    """Configuration for delimited (text table) files.

    :param bool noheader: File has no header and columns the N columns are named ``_1``, ``_2``, ... ``_N`` (0-indexed)

    :param bool impute: Impute column types from the file

    :param comment: Skip lines beginning with the given pattern
    :type comment: str or None

    :param str delimiter: Field delimiter regex

    :param str missing: Specify identifier to be treated as missing

    :param types: Define types of fields in annotations files   
    :type types: str or None
    """
    def __init__(self, noheader=False, impute=False,
                 comment=None, delimiter="\t", missing="NA", types=None):
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

    def to_java(self, hc):
        """Convert to Java TextTableConfiguration object.

        :param :class:`.HailContext` The Hail context.
        """
        return hc.jvm.org.broadinstitute.hail.utils.TextTableConfiguration.apply(self.types, self.comment,
                                                             self.delimiter, self.missing,
                                                             self.noheader, self.impute)
