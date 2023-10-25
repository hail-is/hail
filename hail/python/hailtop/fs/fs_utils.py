from typing import List, Dict
import io
import asyncio

from .. import hail_event_loop
from .router_fs import RouterFS
from .stat_result import FileListEntry


_router_fs_for_loop: Dict[asyncio.AbstractEventLoop, RouterFS] = {}


def _fs() -> RouterFS:
    '''Return an FS for this particular thread.'''
    global _router_fs_for_loop

    loop = hail_event_loop()
    if fs := _router_fs_for_loop.get(loop):
        return fs

    fs = RouterFS()
    _router_fs_for_loop[loop] = fs
    return fs


def open(path: str, mode: str = 'r', buffer_size: int = 8192) -> io.IOBase:
    """Open a file from the local filesystem of from blob storage. Supported
    blob storage providers are GCS, S3 and ABS.

    Examples
    --------
    Write a Pandas DataFrame as a CSV directly into Google Cloud Storage:

    >>> with hfs.open('gs://my-bucket/df.csv', 'w') as f: # doctest: +SKIP
    ...     pandas_df.to_csv(f)

    Read and print the lines of a text file stored in Google Cloud Storage:

    >>> with hfs.open('gs://my-bucket/notes.txt') as f: # doctest: +SKIP
    ...     for line in f:
    ...         print(line.strip())

    Write two lines directly to a file in Google Cloud Storage:

    >>> with hfs.open('gs://my-bucket/notes.txt', 'w') as f: # doctest: +SKIP
    ...     f.write('result1: %s\\n' % result1)
    ...     f.write('result2: %s\\n' % result2)

    Unpack a packed Python struct directly from a file in Google Cloud Storage:

    >>> from struct import unpack
    >>> with hfs.open('gs://my-bucket/notes.txt', 'rb') as f: # doctest: +SKIP
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

    The provided destination file path must be a URI (uniform resource identifier)
    or a path on the local filesystem.

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
    return _fs().open(path, mode, buffer_size)


def copy(src: str, dest: str):
    """Copy a file between filesystems. Filesystems can be local filesystem
    or the blob storage providers GCS, S3 and ABS.

    Examples
    --------
    Copy a file from Google Cloud Storage to a local file:

    >>> hfs.copy('gs://hail-common/LCR.interval_list',
    ...          'file:///mnt/data/LCR.interval_list') # doctest: +SKIP

    Notes
    ----

    If you are copying a file just to then load it into Python, you can use
    :func:`.open` instead. For example:

    >>> with hfs.open('gs://my_bucket/results.csv', 'r') as f: #doctest: +SKIP
    ...     df = pandas_df.read_csv(f)

    The provided source and destination file paths must be URIs
    (uniform resource identifiers) or local filesystem paths.

    Parameters
    ----------
    src: :class:`str`
        Source file URI.
    dest: :class:`str`
        Destination file URI.
    """
    _fs().copy(src, dest)


def exists(path: str) -> bool:
    """Returns ``True`` if `path` exists.

    Parameters
    ----------
    path : :class:`str`

    Returns
    -------
    :obj:`.bool`
    """
    return _fs().exists(path)


def is_file(path: str) -> bool:
    """Returns ``True`` if `path` both exists and is a file.

    Parameters
    ----------
    path : :class:`str`

    Returns
    -------
    :obj:`.bool`
    """
    return _fs().is_file(path)


def is_dir(path: str) -> bool:
    """Returns ``True`` if `path` both exists and is a directory.

    Parameters
    ----------
    path : :class:`str`

    Returns
    -------
    :obj:`.bool`
    """
    return _fs().is_dir(path)


def stat(path: str) -> FileListEntry:
    """Returns information about the file or directory at a given path.

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
    return _fs().stat(path)


def ls(path: str) -> List[FileListEntry]:
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
    return _fs().ls(path)


def mkdir(path: str):
    """Ensure files can be created whose dirname is `path`.

    Warning
    -------

    On file systems without a notion of directories, this function will do nothing. For example,
    on Google Cloud Storage, this operation does nothing.

    """
    _fs().mkdir(path)


def remove(path: str):
    """Removes the file at `path`. If the file does not exist, this function does
    nothing. `path` must be a URI (uniform resource identifier) or a path on the
    local filesystem.

    Parameters
    ----------
    path : :class:`str`
    """
    _fs().remove(path)


def rmtree(path: str):
    """Recursively remove all files under the given `path`. On a local filesystem,
    this removes the directory tree at `path`. On blob storage providers such as
    GCS, S3 and ABS, this removes all files whose name starts with `path`. As such,
    `path` must be a URI (uniform resource identifier) or a path on the local filesystem.

    Parameters
    ----------
    path : :class:`str`
    """
    _fs().rmtree(path)
