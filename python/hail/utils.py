from hail.java import Env, handle_py4j
import io

__all__ = ['hadoop_copy', 'hadoop_read', 'hadoop_write', 'FunctionDocumentation', 'TextTableConfig']


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
def hadoop_read(path, buffer_size=8192):
    """Open a readable file through the Hadoop filesystem API. 
    Supports distributed file systems like hdfs, gs, and s3.
    
    **Example:**
    
    .. doctest::
        :options: +SKIP

        >>> with hadoop_read('gs://my-bucket/notes.txt') as f:
        ...     for line in f:
        ...         print(line.strip())
    
    .. note::
        
        If Spark is running in cluster mode, both the source and destination 
        file paths must be URIs (uniform resource identifiers). This means 
        fully clarified paths, prefixed by scheme (``file://``, ``hdfs://``, ``gs://``,
        ``s3://``, etc.)

    .. caution::
    
        These file handles are slower than standard Python file handles.
        If you are reading a file larger than ~50M, it will be faster to 
        use :py:meth:`~hail.hadoop_copy` to copy the file locally, then read it
        with standard Python I/O tools.
    
    :param str path: Source file URI.
    
    :param int buffer_size: Size of internal buffer.
    
    :return: Iterable file reader.
    :rtype: `io.BufferedReader <https://docs.python.org/2/library/io.html#io.BufferedReader>`_
    """
    if not isinstance(path, str) and not isinstance(path, unicode):
        raise TypeError("expected parameter 'path' to be type str, but found %s" % type(path))
    if not isinstance(buffer_size, int):
        raise TypeError("expected parameter 'buffer_size' to be type int, but found %s" % type(buffer_size))
    return io.BufferedReader(HadoopReader(path), buffer_size=buffer_size)


@handle_py4j
def hadoop_write(path, buffer_size=8192):
    """Open a writable file through the Hadoop filesystem API. 
    Supports distributed file systems like hdfs, gs, and s3.
    
    **Example:**
    
    .. doctest::
        :options: +SKIP

        >>> with hadoop_write('gs://my-bucket/notes.txt') as f:
        ...     f.write('result1: %s\\n' % result1)
        ...     f.write('result2: %s\\n' % result2)
    
    .. note::
        
        If Spark is running in cluster mode, both the source and destination 
        file paths must be URIs (uniform resource identifiers). This means 
        fully clarified paths, prefixed by scheme (``file://``, ``hdfs://``, ``gs://``,
        ``s3://``, etc.)

    .. caution::
    
        These file handles are slower than standard Python file handles. If you
        are writing a large file (larger than ~50M), it will be faster to write
        to a local file using standard Python I/O and use :py:meth:`~hail.hadoop_copy` 
        to move your file to a distributed file system.
    
    :param str path: Destination file URI.
    
    :return: File writer object.
    :rtype: `io.BufferedWriter <https://docs.python.org/2/library/io.html#io.BufferedWriter>`_
    """
    if not isinstance(path, str) and not isinstance(path, unicode):
        raise TypeError("expected parameter 'path' to be type str, but found %s" % type(path))
    if not isinstance(buffer_size, int):
        raise TypeError("expected parameter 'buffer_size' to be type int, but found %s" % type(buffer_size))
    return io.BufferedWriter(HadoopWriter(path), buffer_size=buffer_size)


@handle_py4j
def hadoop_copy(src, dest):
    """Copy a file through the Hadoop filesystem API.
    Supports distributed file systems like hdfs, gs, and s3.
    
    **Example:**
    
    >>> hadoop_copy('gs://hail-common/LCR.interval_list', 'file:///mnt/data/LCR.interval_list') # doctest: +SKIP
    
    .. note::
        
        If Spark is running in cluster mode, both the source and destination 
        file paths must be URIs (uniform resource identifiers). This means 
        fully clarified paths, prefixed by scheme (``file://``, ``hdfs://``, ``gs://``,
        ``s3://``, etc.)
    
    
    :param str src: Source file URI. 
    :param str dest: Destination file URI.
    """
    Env.jutils().copyFile(src, dest, Env.hc()._jhc)


class HadoopReader(io.RawIOBase):
    def __init__(self, path):
        self._jfile = Env.jutils().readFile(path, Env.hc()._jhc)
        super(HadoopReader, self).__init__()

    def close(self):
        self._jfile.close()

    def readable(self):
        return True

    def readinto(self, b):
        b_from_java = self._jfile.read(len(b)).encode('iso-8859-1')
        n_read = len(b_from_java)
        b[:n_read] = b_from_java
        return n_read


class HadoopWriter(io.RawIOBase):
    def __init__(self, path):
        self._jfile = Env.jutils().writeFile(path, Env.hc()._jhc)
        super(HadoopWriter, self).__init__()

    def writable(self):
        return True

    def close(self):
        self._jfile.close()

    def write(self, b):
        self._jfile.write(bytearray(b))
        return len(b)
