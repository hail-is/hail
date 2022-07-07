from typing import List, AsyncContextManager, BinaryIO, Optional
import asyncio
import io
from hailtop.aiotools.local_fs import LocalAsyncFSURL
import nest_asyncio
import os
import functools
import glob
import fnmatch

from hailtop.aiotools.fs import Copier, Transfer, FileListEntry, ReadableStream, WritableStream
from hailtop.aiotools.local_fs import LocalAsyncFS
from hailtop.aiotools.router_fs import RouterAsyncFS
from hailtop.utils import bounded_gather, async_to_blocking

from .fs import FS
from .stat_result import FileType, StatResult


class SyncReadableStream(io.RawIOBase, BinaryIO):  # type: ignore # https://github.com/python/typeshed/blob/a40d79a4e63c4e750a8d3a8012305da942251eb4/stdlib/http/client.pyi#L81
    def __init__(self, ars: ReadableStream, name: str):
        super().__init__()
        self.ars = ars
        self._mode = 'rb'
        self._name = name

    @property
    def mode(self):
        return self._mode

    @property
    def name(self):
        return self._name

    def close(self):
        self.ars.close()
        async_to_blocking(self.ars.wait_closed())

    @property
    def closed(self) -> bool:
        return self.ars.closed

    def fileno(self) -> int:
        raise OSError

    def flush(self):
        pass

    def isatty(self):
        return False

    def readable(self):
        return True

    def seek(self, offset, whence=None):
        raise OSError

    def seekable(self):
        return False

    def tell(self):
        raise io.UnsupportedOperation

    def truncate(self):
        raise io.UnsupportedOperation

    def writable(self):
        return False

    def writelines(self, lines):
        raise OSError

    def read(self, size=-1) -> bytes:
        return async_to_blocking(self.ars.read(size))

    def readall(self) -> bytes:
        return async_to_blocking(self.ars.read(-1))

    def readinto(self, b):
        b[:] = async_to_blocking(self.ars.readexactly(len(b)))

    def write(self, b):
        raise OSError


class SyncWritableStream(io.RawIOBase, BinaryIO):  # type: ignore # https://github.com/python/typeshed/blob/a40d79a4e63c4e750a8d3a8012305da942251eb4/stdlib/http/client.pyi#L81
    def __init__(self, cm: AsyncContextManager[WritableStream], name: str):
        super().__init__()
        self.cm = cm
        self.aws = async_to_blocking(self.cm.__aenter__())
        self._mode = 'wb'
        self._name = name

    @property
    def mode(self):
        return self._mode

    @property
    def name(self):
        return self._name

    def close(self):
        self.aws.close()
        async_to_blocking(self.cm.__aexit__())

    @property
    def closed(self) -> bool:
        return self.aws.closed

    def fileno(self) -> int:
        raise OSError

    def flush(self):
        pass

    def isatty(self):
        return False

    def readable(self):
        return False

    def readline(self, size=-1):
        raise OSError

    def readlines(self, hint=-1):
        raise OSError

    def seek(self, offset, whence=None):
        raise OSError

    def seekable(self):
        return False

    def tell(self):
        raise io.UnsupportedOperation

    def truncate(self):
        raise io.UnsupportedOperation

    def writable(self):
        return True

    def read(self, size=-1):
        raise OSError

    def readall(self):
        raise OSError

    def readinto(self, b):
        raise OSError

    def write(self, b):
        return async_to_blocking(self.aws.write(b))


def _stat_result(is_dir: bool, size_bytes: int, path: str) -> StatResult:
    return StatResult(
        path=path.rstrip('/'),
        size=size_bytes,
        typ=FileType.DIRECTORY if is_dir else FileType.FILE,
        owner=None,
        modification_time=None)


class RouterFS(FS):
    def __init__(self, afs: RouterAsyncFS):
        nest_asyncio.apply()
        self.afs = afs

    def open(self, path: str, mode: str = 'r', buffer_size: int = 8192) -> io.IOBase:
        del buffer_size

        if mode not in ('r', 'rb', 'w', 'wb'):
            raise ValueError(f'Unsupported mode: {repr(mode)}')

        strm: io.IOBase
        if mode[0] == 'r':
            strm = SyncReadableStream(async_to_blocking(self.afs.open(path)), path)
        else:
            assert mode[0] == 'w'
            strm = SyncWritableStream(async_to_blocking(self.afs.create(path)), path)

        if 'b' not in mode:
            strm = io.TextIOWrapper(strm, encoding='utf-8')  # type: ignore # typeshed is wrong, this *is* an IOBase
        return strm

    def copy(self, src: str, dest: str, *, max_simultaneous_transfers=75):
        transfer = Transfer(src, dest)

        async def _copy():
            sema = asyncio.Semaphore(max_simultaneous_transfers)
            await Copier.copy(self.afs, sema, transfer)
        return async_to_blocking(_copy())

    def exists(self, path: str) -> bool:
        async def _exists():
            dir_path = path
            if dir_path[-1] != '/':
                dir_path = dir_path + '/'
            return any(await asyncio.gather(
                self.afs.isfile(path),
                self.afs.isdir(dir_path)))
        return async_to_blocking(_exists())

    def is_file(self, path: str) -> bool:
        return async_to_blocking(self.afs.isfile(path))

    async def _async_is_dir(self, path: str) -> bool:
        if path[-1] != '/':
            path = path + '/'
        return await self.afs.isdir(path)

    def is_dir(self, path: str) -> bool:
        return async_to_blocking(self._async_is_dir(path))

    def stat(self, path: str) -> StatResult:
        size_bytes, is_dir = async_to_blocking(asyncio.gather(
            self._size_bytes_or_none(path), self._async_is_dir(path)))
        if size_bytes is None:
            if not is_dir:
                raise FileNotFoundError(path)
            return _stat_result(True, 0, path)
        return _stat_result(is_dir, size_bytes, path)

    async def _size_bytes_or_none(self, path: str):
        try:
            return await (await self.afs.statfile(path)).size()
        except FileNotFoundError:
            return None

    async def _fle_to_dict(self, fle: FileListEntry) -> StatResult:
        async def size():
            try:
                return await (await fle.status()).size()
            except IsADirectoryError:
                return 0
        return _stat_result(
            *await asyncio.gather(fle.is_dir(), size(), fle.url()))

    def ls(self,
           path: str,
           *,
           error_when_file_and_directory: bool = True,
           _max_simultaneous_files: int = 50) -> List[StatResult]:
        return async_to_blocking(self._async_ls(
            path,
            error_when_file_and_directory=error_when_file_and_directory,
            _max_simultaneous_files=_max_simultaneous_files))

    async def _async_ls(self,
                        path: str,
                        *,
                        error_when_file_and_directory: bool = True,
                        _max_simultaneous_files: int = 50) -> List[StatResult]:
        async def ls_no_glob(path) -> List[StatResult]:
            try:
                return await self._ls_no_glob(path,
                                              error_when_file_and_directory=error_when_file_and_directory,
                                              _max_simultaneous_files=_max_simultaneous_files)
            except FileNotFoundError:
                return []

        url = self.afs.parse_url(path)
        if any(glob.escape(bucket_part) != bucket_part
               for bucket_part in url.bucket_parts):
            raise ValueError(f'glob pattern only allowed in path (e.g. not in bucket): {path}')

        blobpath = url.path
        if isinstance(url, LocalAsyncFSURL) and blobpath[0] != '/':
            blobpath = './' + blobpath

        components = blobpath.split('/')
        assert len(components) > 0

        glob_components = []
        running_prefix = []

        for component in components:
            _raise_for_incomplete_glob_group(component, path)
            if glob.escape(component) == component:
                running_prefix.append(component)
            else:
                glob_components.append((running_prefix, component))
                running_prefix = []

        suffix_components: List[str] = running_prefix
        if len(url.bucket_parts) > 0:
            first_prefix = '/'.join([url.scheme + ':', '', *url.bucket_parts])
        else:
            first_prefix = url.scheme + ':'
        cumulative_prefixes = [first_prefix]

        for intervening_components, single_component_glob_pattern in glob_components:
            cumulative_prefixes = [
                stat.path
                for cumulative_prefix in cumulative_prefixes
                for stat in await ls_no_glob('/'.join([cumulative_prefix, *intervening_components]))
                if fnmatch.fnmatch(stat.path,
                                   '/'.join([cumulative_prefix, *intervening_components, single_component_glob_pattern]))
            ]

        found_stats = [
            stat
            for cumulative_prefix in cumulative_prefixes
            for stat in await ls_no_glob('/'.join([cumulative_prefix, *suffix_components]))
        ]

        if len(glob_components) == 0 and len(found_stats) == 0:
            # Unless we are using a glob pattern, a path referring to no files should error
            raise FileNotFoundError(path)
        return found_stats

    async def _ls_no_glob(self,
                          path: str,
                          *,
                          error_when_file_and_directory: bool = True,
                          _max_simultaneous_files: int = 50) -> List[StatResult]:
        async def ls_as_dir() -> Optional[List[StatResult]]:
            try:
                return await bounded_gather(
                    *[functools.partial(self._fle_to_dict, fle)
                      async for fle in await self.afs.listfiles(path)],
                    parallelism=_max_simultaneous_files)
            except (FileNotFoundError, NotADirectoryError):
                return None
        maybe_size, maybe_contents = await asyncio.gather(
            self._size_bytes_or_none(path), ls_as_dir())

        if maybe_size is not None:
            file_stat = _stat_result(False, maybe_size, path)
            if maybe_contents is not None:
                if error_when_file_and_directory:
                    raise ValueError(f'{path} is both a file and a directory')
                return [file_stat, *maybe_contents]
            return [file_stat]
        if maybe_contents is None:
            raise FileNotFoundError(path)
        return maybe_contents

    def mkdir(self, path: str):
        return async_to_blocking(self.afs.mkdir(path))

    def remove(self, path: str):
        return async_to_blocking(self.afs.remove(path))

    def rmtree(self, path: str):
        return async_to_blocking(self.afs.rmtree(None, path))

    def supports_scheme(self, scheme: str) -> bool:
        return scheme in self.afs.schemes

    def canonicalize_path(self, path: str) -> str:
        if isinstance(self.afs._get_fs(path), LocalAsyncFS):
            if path.startswith('file:'):
                return 'file:' + os.path.realpath(path[5:])
            return 'file:' + os.path.realpath(path)
        return path


def _raise_for_incomplete_glob_group(component: str, full_path: str):
    i = 0
    n = len(component)
    open_group = False
    while i < n:
        c = component[i]
        if c == '[':
            open_group = True
        if c == ']':
            open_group = False
        i += 1
    if open_group:
        raise ValueError(f'glob groups must not include forward slashes: {component} {full_path}')
