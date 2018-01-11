from hail.utils.java import handle_py4j, Env
from hail.typecheck import *
import io

@handle_py4j
@typecheck(path=strlike,
           buffer_size=integral)
def hadoop_read(path, buffer_size=8192):
    """Open a readable file through the Hadoop filesystem API.
    Supports distributed file systems like hdfs, gs, and s3.

    **Examples**

    .. doctest::
        :options: +SKIP

        >>> with hadoop_read('gs://my-bucket/notes.txt') as f:
        ...     for line in f:
        ...         print(line.strip())

    **Notes**

    The provided source file path must be a URI (uniform resource identifier).

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
    return io.BufferedReader(HadoopReader(path), buffer_size=buffer_size)

@handle_py4j
@typecheck(path=strlike,
           buffer_size=integral)
def hadoop_read_binary(path, buffer_size=8192):
    """Open a readable binary file through the Hadoop filesystem API.
    Supports distributed file systems like hdfs, gs, and s3.

    calling :py:meth:`f.read(n_bytes)` on the resulting file handle reads `n_bytes` bytes as a
    Python bytearray. If no argument is provided, the entire file will be read.

    Examples
    --------

    .. doctest::
        :options: +SKIP

        >>> from struct import unpack
        >>> with hadoop_read('gs://my-bucket/notes.txt') as f:
        ...     print(unpack('<f', bytearray(f.read())))

    Notes
    -----

    The provided source file path must be a URI (uniform resource identifier).

    .. caution::

        These file handles are slower than standard Python file handles.
        If you are reading a file larger than ~50M, it will be faster to
        use :py:meth:`~hail.hadoop_copy` to copy the file locally, then read it
        with standard Python I/O tools.

    Parameters
    ----------
    path : :obj:`str`
        Source file URI.
    buffer_size : :obj:`int`
        Size of internal buffer.

    Returns
    -------
    :class:`io.BufferedReader <https://docs.python.org/2/library/io.html#io.BufferedReader>`_
        Binary file reader.
    """
    return io.BufferedReader(HadoopBinaryReader(path), buffer_size=buffer_size)


@handle_py4j
@typecheck(path=strlike,
           buffer_size=integral)
def hadoop_write(path, buffer_size=8192):
    """Open a writable file through the Hadoop filesystem API.
    Supports distributed file systems like hdfs, gs, and s3.

    **Examples**

    .. doctest::
        :options: +SKIP

        >>> with hadoop_write('gs://my-bucket/notes.txt') as f:
        ...     f.write('result1: %s\\n' % result1)
        ...     f.write('result2: %s\\n' % result2)

    **Notes**

    The provided destination file path must be a URI (uniform resource identifier).

    .. caution::

        These file handles are slower than standard Python file handles. If you
        are writing a large file (larger than ~50M), it will be faster to write
        to a local file using standard Python I/O and use :py:meth:`~hail.hadoop_copy`
        to move your file to a distributed file system.

    :param str path: Destination file URI.

    :return: File writer object.
    :rtype: `io.BufferedWriter <https://docs.python.org/2/library/io.html#io.BufferedWriter>`_
    """
    return io.BufferedWriter(HadoopWriter(path), buffer_size=buffer_size)


@handle_py4j
@typecheck(src=strlike,
           dest=strlike)
def hadoop_copy(src, dest):
    """Copy a file through the Hadoop filesystem API.
    Supports distributed file systems like hdfs, gs, and s3.

    **Examples**

    >>> hadoop_copy('gs://hail-common/LCR.interval_list', 'file:///mnt/data/LCR.interval_list') # doctest: +SKIP

    **Notes**

    The provided source and destination file paths must be URIs
    (uniform resource identifiers).

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

class HadoopBinaryReader(io.RawIOBase):
    def __init__(self, path):
        self._jfile = Env.jutils().readBinaryFile(path, Env.hc()._jhc)
        super(HadoopBinaryReader, self).__init__()

    def close(self):
        self._jfile.close()

    def readable(self):
        return True

    def readinto(self, b):
        b_from_java = self._jfile.read(len(b))
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

    def flush(self):
        self._jfile.flush()

    def write(self, b):
        self._jfile.write(bytearray(b))
        return len(b)

