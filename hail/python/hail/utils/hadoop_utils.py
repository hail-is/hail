import gzip
import io
import os.path
import sys
from typing import Any, Dict, List

from deprecated import deprecated

from hail.fs.hadoop_fs import HadoopFS
from hail.typecheck import enumeration, typecheck
from hail.utils.java import Env, info


@deprecated(version="0.2.137", reason="Prefer hailtop.fs.open")
@typecheck(path=str, mode=enumeration('r', 'w', 'x', 'rb', 'wb', 'xb'), buffer_size=int)
def hadoop_open(path: str, mode: str = 'r', buffer_size: int = 8192):
    """Open a file through the Hadoop filesystem API. Supports distributed
    file systems like hdfs, gs, and s3.

    .. deprecated:: 0.2.137
        use :func:`hailtop.fs.open` instead

    Gzip and Deprecation
    --------------------

    This function transparently compresses/decompresses from gzip/bgzip when
    the ``path`` has extenstion ``.gz`` or ``.bgz`` respectively. When using
    the now recommended :func:`hailtop.fs.open`, users will need to implement
    this for themselves. Code such as the following should suffice for reading
    (writing is similar):

    .. code-block:: python3

        import gzip
        import hailtop.fs

        path = ...  # path is gzip data

        with hailtop.fs.open(path, 'rb') as compressed_file:
            with gzip.GzipFile(fileobj=compressed_file, mode='rt') as file:
                ...  # use file here

    See the documentation for the :mod:`gzip` module for more information on
    handling gzip data in python.

    Warning
    -------
    Due to an implementation limitation, :func:`hadoop_open` may be quite
    slow for large data sets (anything larger than 50 MB).

    Examples
    --------
    Write a Pandas DataFrame as a CSV directly into Google Cloud Storage:

    >>> with hadoop_open('gs://my-bucket/df.csv', 'w') as f: # doctest: +SKIP
    ...     pandas_df.to_csv(f)

    Read and print the lines of a text file stored in Google Cloud Storage:

    >>> with hadoop_open('gs://my-bucket/notes.txt') as f: # doctest: +SKIP
    ...     for line in f:
    ...         print(line.strip())

    Write two lines directly to a file in Google Cloud Storage:

    >>> with hadoop_open('gs://my-bucket/notes.txt', 'w') as f: # doctest: +SKIP
    ...     f.write('result1: %s\\n' % result1)
    ...     f.write('result2: %s\\n' % result2)

    Unpack a packed Python struct directly from a file in Google Cloud Storage:

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
    path : :class:`str`
        Path to file.
    mode : :class:`str`
        File access mode.
    buffer_size : :obj:`int`
        Buffer size, in bytes.

    Returns
    -------
        Readable or writable file handle.
    """
    # pile of hacks to preserve some legacy behavior, like auto gzip
    fs = Env.fs()
    if isinstance(fs, HadoopFS):
        return fs.legacy_open(path, mode, buffer_size)
    _, ext = os.path.splitext(path)
    if ext in ('.gz', '.bgz'):
        binary_mode = mode[0] + 'b'
        file = fs.open(path, binary_mode, buffer_size)
        file = gzip.GzipFile(fileobj=file, mode=mode)
        if 'b' not in mode:
            file = io.TextIOWrapper(file, encoding='utf-8')
    else:
        file = fs.open(path, mode, buffer_size)
    return file


@deprecated(version="0.2.137", reason="Prefer hailtop.fs.copy")
@typecheck(src=str, dest=str)
def hadoop_copy(src, dest):
    """Copy a file through the Hadoop filesystem API.
    Supports distributed file systems like hdfs, gs, and s3.

    .. deprecated:: 0.2.137
        use :func:`hailtop.fs.copy` instead

    Examples
    --------
    Copy a file from Google Cloud Storage to a local file:

    >>> hadoop_copy('gs://hail-common/LCR.interval_list',
    ...             'file:///mnt/data/LCR.interval_list') # doctest: +SKIP

    Notes
    ----

    Try using :func:`.hadoop_open` first, it's simpler, but not great
    for large data! For example:

    >>> with hadoop_open('gs://my_bucket/results.csv', 'r') as f: #doctest: +SKIP
    ...     pandas_df.to_csv(f)

    The provided source and destination file paths must be URIs
    (uniform resource identifiers).

    Parameters
    ----------
    src: :class:`str`
        Source file URI.
    dest: :class:`str`
        Destination file URI.
    """
    return Env.fs().copy(src, dest)


@deprecated(version="0.2.137", reason="Prefer hailtop.fs.exists")
def hadoop_exists(path: str) -> bool:
    """Returns ``True`` if `path` exists.

    .. deprecated:: 0.2.137
        use :func:`hailtop.fs.exists` instead

    Parameters
    ----------
    path : :class:`str`

    Returns
    -------
    :obj:`.bool`
    """
    return Env.fs().exists(path)


@deprecated(version="0.2.137", reason="Prefer hailtop.fs.is_file")
def hadoop_is_file(path: str) -> bool:
    """Returns ``True`` if `path` both exists and is a file.

    .. deprecated:: 0.2.137
        use :func:`hailtop.fs.is_file` instead

    Parameters
    ----------
    path : :class:`str`

    Returns
    -------
    :obj:`.bool`
    """
    return Env.fs().is_file(path)


@deprecated(version="0.2.137", reason="Prefer hailtop.fs.is_dir")
def hadoop_is_dir(path: str) -> bool:
    """Returns ``True`` if `path` both exists and is a directory.

    .. deprecated:: 0.2.137
        use :func:`hailtop.fs.is_dir` instead

    Parameters
    ----------
    path : :class:`str`

    Returns
    -------
    :obj:`.bool`
    """
    return Env.fs().is_dir(path)


@deprecated(version="0.2.137", reason="Prefer hailtop.fs.stat")
def hadoop_stat(path: str) -> Dict[str, Any]:
    """Returns information about the file or directory at a given path.

    .. deprecated:: 0.2.137
        use :func:`hailtop.fs.stat` instead

    Notes
    -----
    Raises an error if `path` does not exist.

    The resulting dictionary contains the following data:

    - is_dir (:obj:`bool`) -- Path is a directory.
    - size_bytes (:obj:`int`) -- Size in bytes.
    - size (:class:`str`) -- Size as a readable string.
    - modification_time (:class:`str`) -- Time of last file modification.
    - owner (:class:`str`) -- Owner.
    - path (:class:`str`) -- Path.

    Parameters
    ----------
    path : :class:`str`

    Returns
    -------
    :obj:`dict`
    """
    return Env.fs().stat(path).to_legacy_dict()


@deprecated(version="0.2.137", reason="Prefer hailtop.fs.ls")
def hadoop_ls(path: str) -> List[Dict[str, Any]]:
    """Returns information about files at `path`.

    .. deprecated:: 0.2.137
        use :func:`hailtop.fs.ls` instead

    Notes
    -----
    Raises an error if `path` does not exist.

    If `path` is a file, returns a list with one element. If `path` is a
    directory, returns an element for each file contained in `path` (does not
    search recursively).

    Each dict element of the result list contains the following data:

    - is_dir (:obj:`bool`) -- Path is a directory.
    - size_bytes (:obj:`int`) -- Size in bytes.
    - size (:class:`str`) -- Size as a readable string.
    - modification_time (:class:`str`) -- Time of last file modification.
    - owner (:class:`str`) -- Owner.
    - path (:class:`str`) -- Path.

    Parameters
    ----------
    path : :class:`str`

    Returns
    -------
    :obj:`list` [:obj:`dict`]
    """
    return [sr.to_legacy_dict() for sr in Env.fs().ls(path)]


def hadoop_scheme_supported(scheme: str) -> bool:
    """Returns ``True`` if the Hadoop filesystem supports URLs with the given
    scheme.

    Examples
    --------

    >>> hadoop_scheme_supported('gs') # doctest: +SKIP

    Notes
    -----
    URLs with the `https` scheme are only supported if they are specifically
    Azure Blob Storage URLs of the form `https://<ACCOUNT_NAME>.blob.core.windows.net/<CONTAINER_NAME>/<PATH>`

    Parameters
    ----------
    scheme : :class:`str`

    Returns
    -------
    :obj:`.bool`
    """
    return Env.fs().supports_scheme(scheme)


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
    path: :class:`str`
    """
    from hail.utils import local_path_uri

    log = os.path.realpath(Env.hc()._log)
    try:
        if hadoop_is_dir(path):
            _, tail = os.path.split(log)
            path = os.path.join(path, tail)
        info(f"copying log to {path!r}...")
        hadoop_copy(local_path_uri(log), path)
    except Exception as e:
        sys.stderr.write(f'Could not copy log: encountered error:\n  {e}')
