from py4j.protocol import Py4JJavaError
from hail.java import env, raise_py4j_exception


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

    :ivar bool noheader: File has no header and columns the N columns are named ``_1``, ``_2``, ... ``_N`` (0-indexed)
    :ivar bool impute: Impute column types from the file
    :ivar comment: Skip lines beginning with the given pattern
    :vartype comment: str or None
    :ivar str delimiter: Field delimiter regex
    :ivar str missing: Specify identifier to be treated as missing
    :ivar types: Define types of fields in annotations files
    :vartype types: str or None
    """

    def __init__(self, noheader=False, impute=False,
                 comment=None, delimiter="\\t", missing="NA", types=None):
        self.noheader = noheader
        self.impute = impute
        self.comment = comment
        self.delimiter = delimiter
        self.missing = missing
        self.types = types

    def __str__(self):
        return self._to_java().toString()

    def _to_java(self):
        """Convert to Java TextTableConfiguration object."""
        try:
            return env.hail.utils.TextTableConfiguration.apply(self.types, self.comment,
                                                               self.delimiter, self.missing,
                                                               self.noheader, self.impute)
        except Py4JJavaError as e:
            raise_py4j_exception(e)
