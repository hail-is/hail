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
    """Open an iterable file handle. Supports distributed file systems like hdfs, gs, and s3.
    
    .. doctest::
    :options: +SKIP

        >>> with hdfs_read('gs://my-bucket/notes.txt') as f:
        ...     for line in f:
        ...         print(line)
    
    .. caution::
    
        These file handles are *much* slower than standard python file handles.
        If you are reading a file larger than a few thousand lines, it may be 
        faster to use :py:meth`.hdfs_copy` to copy the file locally, then read it
        with standard Python I/O tools.

    
    :param str path: Source file.
    
    :return: Iterable file reader object.
    :rtype: :class:`.HadoopReader`
    """
    return HadoopReader(path)


@handle_py4j
def hdfs_write(path):
    """Open a writable file handle. Supports distributed file systems like hdfs, gs, and s3. 
    
    .. doctest::
    :options: +SKIP

        >>> with hdfs_write('gs://my-bucket/notes.txt') as f:
        ...     f.write('result1: %s' % result1)
        ...     f.write('result2: %s' % result2)
    
    .. caution::
    
        These file handles are *much* slower than standard python file handles.
        It may be faster to write to a local file using standard Python I/O and 
        use :py:meth`.hdfs_copy` to move your file to a distributed file system.
    
    :param str path: Destination file.
    
    :return: File writer object.
    :rtype: :class:`.HadoopWriter:
    """
    return HadoopWriter(path)


@handle_py4j
def hdfs_copy(src, dest):
    """Copy a file. Supports distributed file systems like hdfs, gs, and s3.
    
    >>> hdfs_copy('gs://hail-common/LCR.interval_list', 'file:///user/me/LCR.interval_list') # doctest: +SKIP
    
    :param str src: Source file. 
    :param str dest: Destination file.
    """
    Env.jutils().copyFile(src, dest, Env.hc()._jhc)


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
