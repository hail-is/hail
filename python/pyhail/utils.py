from py4j.protocol import Py4JJavaError

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
                 comment=None, delimiter="\\t", missing="NA", types=None):
        self.noheader = noheader
        self.impute = impute
        self.comment = comment
        self.delimiter = delimiter
        self.missing = missing
        self.types = types

    def as_pargs(self):
        """Configuration parameters as a list"""

        pargs = ["--delimiter", self.delimiter,
               "--missing", self.missing]

        if self.noheader:
            pargs.append("--no-header")

        if self.impute:
            pargs.append("--impute")

        if self.types:
            pargs.append("--types")
            pargs.append(self.types)

        if self.comment:
            pargs.append("--comment")
            pargs.append(self.comment)

        return pargs

    def __str__(self):
        return " ".join(self.as_pargs())

    def to_java(self, hc):
        """Convert to Java TextTableConfiguration object.

        :param :class:`.HailContext` The Hail context.
        """
        try:
            return hc.jvm.org.broadinstitute.hail.utils.TextTableConfiguration.apply(self.types, self.comment,
                                                             self.delimiter, self.missing,
                                                             self.noheader, self.impute)
        except Py4JJavaError as e:
            self._raise_py4j_exception(e)
