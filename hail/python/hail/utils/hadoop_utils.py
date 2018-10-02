from hail.utils.java import Env, info
from hail.utils import local_path_uri
from hail.typecheck import *
import io
import json
import os
import sys
from typing import Dict, List

@typecheck(path=str,
           mode=enumeration('r', 'w', 'x', 'rb', 'wb', 'xb'),
           buffer_size=int)
def hadoop_open(path: str, mode: str = 'r', buffer_size: int = 8192):
    """Open a file through the Hadoop filesystem API. Supports distributed
    file systems like hdfs, gs, and s3.

    Examples
    --------

    >>> with hadoop_open('gs://my-bucket/notes.txt') as f: # doctest: +SKIP
    ...     for line in f:
    ...         print(line.strip())

    >>> with hadoop_open('gs://my-bucket/notes.txt', 'w') as f: # doctest: +SKIP
    ...     f.write('result1: %s\\n' % result1)
    ...     f.write('result2: %s\\n' % result2)

    >>> from struct import unpack
    >>> with hadoop_open('gs://my-bucket/notes.txt', 'rb') as f: # doctest: +SKIP
    ...     print(unpack('<f', bytearray(f.read())))

    Notes
    -----
    The supported modes are:

     - ``'r'`` -- Readable text file (:class:`io.TextIOWrapper`). Default behavior.
     - ``'w'`` -- Writable text file (:class:`io.TextIOWrapper`).
     - ``'x'`` -- Exclusive writable text file (:class:`io.TextIOWrapper`).
       Throws an error if a file already exists at the path.
     - ``'rb'`` -- Readable binary file (:class:`io.BufferedReader`).
     - ``'wb'`` -- Writable binary file (:class:`io.BufferedWriter`).
     - ``'xb'`` -- Exclusive writable binary file (:class:`io.BufferedWriter`).
       Throws an error if a file already exists at the path.


    The provided destination file path must be a URI (uniform resource identifier).

    .. caution::

        These file handles are slower than standard Python file handles. If you
        are writing a large file (larger than ~50M), it will be faster to write
        to a local file using standard Python I/O and use :func:`.hadoop_copy`
        to move your file to a distributed file system.

    Parameters
    ----------
    path : :obj:`str`
        Path to file.
    mode : :obj:`str`
        File access mode.
    buffer_size : :obj:`int`
        Buffer size, in bytes.

    Returns
    -------
        Readable or writable file handle.
    """
    if 'r' in mode:
        handle = io.BufferedReader(HadoopReader(path, buffer_size), buffer_size=buffer_size)
    elif 'w' in mode:
        handle = io.BufferedWriter(HadoopWriter(path), buffer_size=buffer_size)
    elif 'x' in mode:
        handle = io.BufferedWriter(HadoopWriter(path, exclusive=True), buffer_size=buffer_size)

    if 'b' in mode:
        return handle
    else:
        return io.TextIOWrapper(handle, encoding='iso-8859-1')


@typecheck(src=str,
           dest=str)
def hadoop_copy(src, dest):
    """Copy a file through the Hadoop filesystem API.
    Supports distributed file systems like hdfs, gs, and s3.

    Examples
    --------

    >>> hadoop_copy('gs://hail-common/LCR.interval_list', 'file:///mnt/data/LCR.interval_list') # doctest: +SKIP

    Notes
    -----
    The provided source and destination file paths must be URIs
    (uniform resource identifiers).

    Parameters
    ----------
    src: :obj:`str`
        Source file URI.
    dest: :obj:`str`
        Destination file URI.
    """
    Env.jutils().copyFile(src, dest, Env.hc()._jhc)


class HadoopReader(io.RawIOBase):
    def __init__(self, path, buffer_size):
        self._jfile = Env.jutils().readFile(path, Env.hc()._jhc, buffer_size)
        super(HadoopReader, self).__init__()

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
    def __init__(self, path, exclusive=False):
        self._jfile = Env.jutils().writeFile(path, Env.hc()._jhc, exclusive)
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


def hadoop_exists(path: str) -> bool:
    """Returns ``True`` if `path` exists.

    Parameters
    ----------
    path : :obj:`str`

    Returns
    -------
    :obj:`.bool`
    """
    return Env.jutils().exists(path, Env.hc()._jhc)


def hadoop_is_file(path: str) -> bool:
    """Returns ``True`` if `path` both exists and is a file.

    Parameters
    ----------
    path : :obj:`str`

    Returns
    -------
    :obj:`.bool`
    """
    return Env.jutils().isFile(path, Env.hc()._jhc)


def hadoop_is_dir(path) -> bool:
    """Returns ``True`` if `path` both exists and is a directory.

    Parameters
    ----------
    path : :obj:`str`

    Returns
    -------
    :obj:`.bool`
    """
    return Env.jutils().isDir(path, Env.hc()._jhc)


def hadoop_stat(path: str) -> Dict:
    """Returns information about the file or directory at a given path.

    Notes
    -----
    Raises an error if `path` does not exist.

    The resulting dictionary contains the following data:

    - is_dir (:obj:`bool`) -- Path is a directory.
    - size_bytes (:obj:`int`) -- Size in bytes.
    - size (:obj:`str`) -- Size as a readable string.
    - modification_time (:obj:`str`) -- Time of last file modification.
    - owner (:obj:`str`) -- Owner.
    - path (:obj:`str`) -- Path.

    Parameters
    ----------
    path : :obj:`str`

    Returns
    -------
    :obj:`Dict`
    """
    return json.loads(Env.jutils().stat(path, Env.hc()._jhc))


def hadoop_ls(path: str) -> List[Dict]:
    """Returns information about files at `path`.

    Notes
    -----
    Raises an error if `path` does not exist.

    If `path` is a file, returns a list with one element. If `path` is a
    directory, returns an element for each file contained in `path` (does not
    search recursively).

    Each dict element of the result list contains the following data:

    - is_dir (:obj:`bool`) -- Path is a directory.
    - size_bytes (:obj:`int`) -- Size in bytes.
    - size (:obj:`str`) -- Size as a readable string.
    - modification_time (:obj:`str`) -- Time of last file modification.
    - owner (:obj:`str`) -- Owner.
    - path (:obj:`str`) -- Path.

    Parameters
    ----------
    path : :obj:`str`

    Returns
    -------
    :obj:`List[Dict]`
    """
    r = Env.jutils().ls(path, Env.hc()._jhc)
    return json.loads(r)


def copy_log(path: str) -> None:
    """Attempt to copy the session log to a hadoop-API-compatible location.

    Examples
    --------
    Specify a manual path:

    >>> hl.copy_log('gs://my-bucket/analysis-10-jan19.log')  # doctest: +SKIP
    INFO: copying log to 'gs://my-bucket/analysis-10-jan19.log'...

    Copy to a directory:

    >>> hl.copy_log('gs://my-bucket/')  # doctest: +SKIP
    INFO: copying log to 'gs://my-bucket/hail-20180924-2018-devel-46e5fad57524.log'...

    Notes
    -----
    Since Hail cannot currently log directly to distributed file systems, this
    function is provided as a utility for offloading logs from ephemeral nodes.

    If `path` is a directory, then the log file will be copied using its
    base name to the directory (e.g. ``/home/hail.log`` would be copied as
    ``gs://my-bucket/hail.log`` if `path` is ``gs://my-bucket``.

    Parameters
    ----------
    path: :obj:`str`
    """
    log = Env.hc()._log
    try:
        if hadoop_is_dir(path):
            _, tail = os.path.split(log)
            path = os.path.join(path, tail)
        info(f"copying log to {repr(path)}...")
        hadoop_copy(local_path_uri(Env.hc()._log), path)
    except Exception as e:
        sys.stderr.write(f'Could not copy log: encountered error:\n  {e}')