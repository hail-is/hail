from typing import List, AsyncContextManager, BinaryIO, Optional, Tuple, Dict, Any
import asyncio
import io
import os
import functools
import glob
import fnmatch

from hailtop.aiotools.fs import (
    Copier,
    Transfer,
    FileListEntry as AIOFileListEntry,
    ReadableStream,
    WritableStream,
    AsyncFSURL,
)
from hailtop.aiotools.router_fs import RouterAsyncFS
from hailtop.utils import bounded_gather2, async_to_blocking

from .fs import FS
from .stat_result import FileType, FileListEntry


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

    def seek(self, offset: int, whence: int = os.SEEK_SET) -> int:
        return async_to_blocking(self.ars.seek(offset, whence))

    def seekable(self) -> bool:
        return self.ars.seekable()

    def tell(self) -> int:
        return self.ars.tell()

    def truncate(self, size: Optional[int] = None):
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
        async_to_blocking(self.cm.__aexit__(None, None, None))

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

    def readline(self, size: Optional[int] = -1):
        raise OSError

    def readlines(self, hint=-1):
        raise OSError

    def seek(self, offset, whence=None):
        raise OSError

    def seekable(self):
        return False

    def tell(self):
        raise io.UnsupportedOperation

    def truncate(self, size: Optional[int] = None):
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


def _stat_result(is_dir: bool, size_bytes_and_time_modified: Optional[Tuple[int, float]], path: str) -> FileListEntry:
    if size_bytes_and_time_modified:
        size_bytes, time_modified = size_bytes_and_time_modified
    else:
        size_bytes = 0
        time_modified = None

    return FileListEntry(
        path=path.rstrip('/'),
        size=size_bytes,
        typ=FileType.DIRECTORY if is_dir else FileType.FILE,
        owner=None,
        modification_time=time_modified,
    )


class RouterFS(FS):
    def __init__(
        self,
        afs: Optional[RouterAsyncFS] = None,
        *,
        local_kwargs: Optional[Dict[str, Any]] = None,
        gcs_kwargs: Optional[Dict[str, Any]] = None,
        azure_kwargs: Optional[Dict[str, Any]] = None,
        s3_kwargs: Optional[Dict[str, Any]] = None,
    ):
        if afs and (local_kwargs or gcs_kwargs or azure_kwargs or s3_kwargs):
            raise ValueError(
                f'If afs is specified, no other arguments may be specified: {afs=}, {local_kwargs=}, {gcs_kwargs=}, {azure_kwargs=}, {s3_kwargs=}'
            )
        self.afs = afs or RouterAsyncFS(
            local_kwargs=local_kwargs, gcs_kwargs=gcs_kwargs, azure_kwargs=azure_kwargs, s3_kwargs=s3_kwargs
        )

    @property
    def _gcs_kwargs(self) -> Optional[Dict[str, Any]]:
        return self.afs._gcs_kwargs

    def open(self, path: str, mode: str = 'r', buffer_size: int = 8192) -> io.IOBase:
        del buffer_size

        if mode not in ('r', 'rb', 'w', 'wb'):
            raise ValueError(f'Unsupported mode: {mode!r}')

        strm: io.IOBase
        if mode[0] == 'r':
            strm = SyncReadableStream(async_to_blocking(self.afs.open(path)), path)
        else:
            assert mode[0] == 'w'
            try:
                async_strm = async_to_blocking(self.afs.create(path))
            except (FileNotFoundError, NotADirectoryError):
                async_to_blocking(self.afs.makedirs(os.path.dirname(path)))
                async_strm = async_to_blocking(self.afs.create(path))
            strm = SyncWritableStream(async_strm, path)

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
            return any(await asyncio.gather(self.afs.isfile(path), self.afs.isdir(dir_path)))

        return async_to_blocking(_exists())

    def is_file(self, path: str) -> bool:
        return async_to_blocking(self.afs.isfile(path))

    async def _async_is_dir(self, path: str) -> bool:
        if path[-1] != '/':
            path = path + '/'
        return await self.afs.isdir(path)

    def is_dir(self, path: str) -> bool:
        return async_to_blocking(self._async_is_dir(path))

    def stat(self, path: str) -> FileListEntry:
        maybe_sb_and_t, is_dir = async_to_blocking(
            asyncio.gather(self._size_bytes_and_time_modified_or_none(path), self._async_is_dir(path))
        )
        if maybe_sb_and_t is None:
            if not is_dir:
                raise FileNotFoundError(path)
            return _stat_result(True, None, path)
        return _stat_result(is_dir, maybe_sb_and_t, path)

    async def _size_bytes_and_time_modified_or_none(self, path: str) -> Optional[Tuple[int, float]]:
        try:
            # Hadoop semantics: creation time is used if the object has no notion of last modification time.
            file_status = await self.afs.statfile(path)
            return (await file_status.size(), file_status.time_modified().timestamp())
        except FileNotFoundError:
            return None

    async def _aiofle_to_fle(self, fle: AIOFileListEntry) -> FileListEntry:
        async def maybe_status() -> Optional[Tuple[int, float]]:
            try:
                file_status = await fle.status()
                return (await file_status.size(), file_status.time_modified().timestamp())
            except IsADirectoryError:
                return None

        return _stat_result(*await asyncio.gather(fle.is_dir(), maybe_status(), fle.url()))

    def ls(
        self, path: str, *, error_when_file_and_directory: bool = True, _max_simultaneous_files: int = 50
    ) -> List[FileListEntry]:
        return async_to_blocking(
            self._async_ls(
                path,
                error_when_file_and_directory=error_when_file_and_directory,
                _max_simultaneous_files=_max_simultaneous_files,
            )
        )

    async def _async_ls(
        self, path: str, *, error_when_file_and_directory: bool = True, _max_simultaneous_files: int = 50
    ) -> List[FileListEntry]:
        sema = asyncio.Semaphore(_max_simultaneous_files)

        async def ls_no_glob(path) -> List[FileListEntry]:
            try:
                return await self._ls_no_glob(
                    path, error_when_file_and_directory=error_when_file_and_directory, sema=sema
                )
            except FileNotFoundError:
                return []

        async def list_within_each_prefix(prefixes: List[AsyncFSURL], parts: List[str]) -> List[List[FileListEntry]]:
            pfs = [functools.partial(ls_no_glob, str(prefix.with_new_path_components(*parts))) for prefix in prefixes]
            return await bounded_gather2(sema, *pfs, cancel_on_error=True)

        url = self.afs.parse_url(path)
        if any(glob.escape(bucket_part) != bucket_part for bucket_part in url.bucket_parts):
            raise ValueError(f'glob pattern only allowed in path (e.g. not in bucket): {path}')

        blobpath = url.path
        if blobpath == '':
            components = []
        else:
            components = blobpath.split('/')

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
        cached_stats_for_each_cumulative_prefix: Optional[List[FileListEntry]] = None
        cumulative_prefixes: List[AsyncFSURL] = [url.with_path('')]

        for intervening_components, single_component_glob_pattern in glob_components:
            stats_grouped_by_prefix = await list_within_each_prefix(cumulative_prefixes, intervening_components)
            cached_stats_for_each_cumulative_prefix = [
                stat
                for stats_for_one_prefix, cumulative_prefix in zip(stats_grouped_by_prefix, cumulative_prefixes)
                for stat in stats_for_one_prefix
                if fnmatch.fnmatch(
                    stat.path,
                    str(
                        cumulative_prefix.with_new_path_components(
                            *intervening_components, single_component_glob_pattern
                        )
                    ),
                )
            ]
            cumulative_prefixes = [self.afs.parse_url(stat.path) for stat in cached_stats_for_each_cumulative_prefix]

        if len(suffix_components) == 0 and cached_stats_for_each_cumulative_prefix is not None:
            found_stats = cached_stats_for_each_cumulative_prefix
        else:
            found_stats_grouped_by_prefix = await list_within_each_prefix(cumulative_prefixes, suffix_components)
            found_stats = [stat for stats in found_stats_grouped_by_prefix for stat in stats]

        if len(glob_components) == 0 and len(found_stats) == 0:
            # Unless we are using a glob pattern, a path referring to no files should error
            raise FileNotFoundError(path)
        return found_stats

    async def _ls_no_glob(
        self, path: str, *, error_when_file_and_directory: bool = True, sema: asyncio.Semaphore
    ) -> List[FileListEntry]:
        async def ls_as_dir() -> Optional[List[FileListEntry]]:
            try:
                return await bounded_gather2(
                    sema,
                    *[functools.partial(self._aiofle_to_fle, fle) async for fle in await self.afs.listfiles(path)],
                    cancel_on_error=True,
                )
            except (FileNotFoundError, NotADirectoryError):
                return None

        maybe_sb_and_t, maybe_contents = await asyncio.gather(
            self._size_bytes_and_time_modified_or_none(path), ls_as_dir()
        )

        if maybe_sb_and_t is not None:
            file_stat = _stat_result(False, maybe_sb_and_t, path)
            if maybe_contents is not None:
                if error_when_file_and_directory:
                    raise ValueError(f'{path} is both a file and a directory')
                return [file_stat, *maybe_contents]
            return [file_stat]
        if maybe_contents is None:
            raise FileNotFoundError(path)
        return maybe_contents

    def mkdir(self, path: str) -> None:
        return async_to_blocking(self.afs.mkdir(path))

    async def amkdir(self, path: str) -> None:
        return await self.afs.mkdir(path)

    def remove(self, path: str) -> None:
        return async_to_blocking(self.afs.remove(path))

    async def aremove(self, path: str) -> None:
        return await self.afs.remove(path)

    def rmtree(self, path: str) -> None:
        return async_to_blocking(self.afs.rmtree(None, path))

    async def armtree(self, path: str) -> None:
        return await self.afs.rmtree(None, path)

    def supports_scheme(self, scheme: str) -> bool:
        return scheme in self.afs.schemes()

    def canonicalize_path(self, path: str) -> str:
        url = self.afs.parse_url(path)
        if url.scheme == 'file':
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
