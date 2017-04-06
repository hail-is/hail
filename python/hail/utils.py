from hail.java import Env, handle_py4j

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

    @handle_py4j
    def _to_java(self):
        """Convert to Java TextTableConfiguration object."""
        return Env.hail().utils.TextTableConfiguration.apply(self.types, self.comment,
                                                           self.delimiter, self.missing,
                                                           self.noheader, self.impute)

class FunctionDocumentation(object):

    @handle_py4j
    def types_rst(self, file_name):
        Env.hail().utils.FunctionDocumentation.makeTypesDocs(file_name)

    @handle_py4j
    def functions_rst(self, file_name):
        Env.hail().utils.FunctionDocumentation.makeFunctionsDocs(file_name)

@handle_py4j
def hdfs_read(path):
    return HadoopReader(path)

@handle_py4j
def hdfs_write(path):
    return HadoopWriter(path)

class HadoopReader(object):
    def __init__(self, path):
        self._jfile = Env.jutils().readFile(path, Env.hc()._jhc)

    def __iter__(self):
        return self

    @handle_py4j
    def next(self):
        if not self._jfile.hasNext():
            raise StopIteration
        else:
            return self._jfile.next()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._jfile.close()

    def close(self):
        self._jfile.close()

class HadoopWriter(object):
    def __init__(self, path):
        self._jfile = Env.jutils().writeFile(path, Env.hc()._jhc)

    @handle_py4j
    def write(self, text, newline=True):
        if newline:
            self._jfile.writeLine(text)
        else:
            self._jfile.write(text)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._jfile.close()

    def close(self):
        self._jfile.close()