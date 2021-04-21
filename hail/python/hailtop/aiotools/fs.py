from typing import Any, Optional, List, Type, BinaryIO, cast, Set, AsyncIterator, Union, Dict
from types import TracebackType
import abc
import os
import os.path
import io
import stat
import shutil
import asyncio
from concurrent.futures import ThreadPoolExecutor
import urllib.parse
import humanize
from hailtop.utils import (
    retry_transient_errors, blocking_to_async, url_basename, url_join, bounded_gather2,
    time_msecs, humanize_timedelta_msecs)
from .stream import ReadableStream, WritableStream, blocking_readable_stream_to_async, blocking_writable_stream_to_async


class FileStatus(abc.ABC):
    @abc.abstractmethod
    async def size(self) -> int:
        pass

    @abc.abstractmethod
    async def __getitem__(self, key: str) -> Any:
        pass


class FileListEntry(abc.ABC):
    @abc.abstractmethod
    def name(self) -> str:
        pass

    @abc.abstractmethod
    async def url(self) -> str:
        pass

    @abc.abstractmethod
    def url_maybe_trailing_slash(self) -> str:
        pass

    @abc.abstractmethod
    async def is_file(self) -> bool:
        pass

    @abc.abstractmethod
    async def is_dir(self) -> bool:
        pass

    @abc.abstractmethod
    async def status(self) -> FileStatus:
        pass


class MultiPartCreate(abc.ABC):
    @abc.abstractmethod
    async def create_part(self, number: int, start: int, *, retry_writes: bool = True):
        pass

    @abc.abstractmethod
    async def __aenter__(self) -> 'MultiPartCreate':
        pass

    @abc.abstractmethod
    async def __aexit__(self,
                        exc_type: Optional[Type[BaseException]],
                        exc_val: Optional[BaseException],
                        exc_tb: Optional[TracebackType]) -> None:
        pass


class AsyncFS(abc.ABC):
    FILE = 'file'
    DIR = 'dir'

    @abc.abstractmethod
    def schemes(self) -> Set[str]:
        pass

    @abc.abstractmethod
    async def open(self, url: str) -> ReadableStream:
        pass

    @abc.abstractmethod
    async def open_from(self, url: str, start: int) -> ReadableStream:
        pass

    @abc.abstractmethod
    async def create(self, url: str, *, retry_writes: bool = True) -> WritableStream:
        pass

    @abc.abstractmethod
    async def multi_part_create(
            self,
            sema: asyncio.Semaphore,
            url: str,
            num_parts: int) -> MultiPartCreate:
        pass

    @abc.abstractmethod
    async def mkdir(self, url: str) -> None:
        pass

    @abc.abstractmethod
    async def makedirs(self, url: str, exist_ok: bool = False) -> None:
        pass

    @abc.abstractmethod
    async def statfile(self, url: str) -> FileStatus:
        pass

    @abc.abstractmethod
    async def listfiles(self, url: str, recursive: bool = False) -> AsyncIterator[FileListEntry]:
        pass

    @abc.abstractmethod
    async def staturl(self, url: str) -> str:
        pass

    @abc.abstractmethod
    async def isfile(self, url: str) -> bool:
        pass

    @abc.abstractmethod
    async def isdir(self, url: str) -> bool:
        pass

    @abc.abstractmethod
    async def remove(self, url: str) -> None:
        pass

    @abc.abstractmethod
    async def rmtree(self, sema: asyncio.Semaphore, url: str) -> None:
        pass

    async def touch(self, url: str) -> None:
        async with await self.create(url):
            pass

    async def close(self) -> None:
        pass

    async def __aenter__(self) -> 'AsyncFS':
        return self

    async def __aexit__(self,
                        exc_type: Optional[Type[BaseException]],
                        exc_val: Optional[BaseException],
                        exc_tb: Optional[TracebackType]) -> None:
        await self.close()


class LocalStatFileStatus(FileStatus):
    def __init__(self, stat_result):
        self._stat_result = stat_result
        self._items = None

    async def size(self) -> int:
        return self._stat_result.st_size

    async def __getitem__(self, key: str) -> Any:
        raise KeyError(key)


class LocalFileListEntry(FileListEntry):
    def __init__(self, thread_pool, base_url, entry):
        assert '/' not in entry.name
        self._thread_pool = thread_pool
        if not base_url.endswith('/'):
            base_url = f'{base_url}/'
        self._base_url = base_url
        self._entry = entry
        self._status = None

    def name(self) -> str:
        return self._entry.name

    async def url(self) -> str:
        trailing_slash = "/" if await self.is_dir() else ""
        return f'{self._base_url}{self._entry.name}{trailing_slash}'

    def url_maybe_trailing_slash(self) -> str:
        return f'{self._base_url}{self._entry.name}'

    async def is_file(self) -> bool:
        return not await self.is_dir()

    async def is_dir(self) -> bool:
        return await blocking_to_async(self._thread_pool, self._entry.is_dir)

    async def status(self) -> LocalStatFileStatus:
        if self._status is None:
            if await self.is_dir():
                raise ValueError("directory has no file status")
            self._status = LocalStatFileStatus(await blocking_to_async(self._thread_pool, self._entry.stat))
        return self._status


class LocalMultiPartCreate(MultiPartCreate):
    def __init__(self, fs: 'LocalAsyncFS', path: str, num_parts: int):
        self._fs = fs
        self._path = path
        self._num_parts = num_parts

    async def create_part(self, number: int, start: int, *, retry_writes: bool = True):  # pylint: disable=unused-argument
        assert 0 <= number < self._num_parts
        f = await blocking_to_async(self._fs._thread_pool, open, self._path, 'r+b')
        f.seek(start)
        return blocking_writable_stream_to_async(self._fs._thread_pool, cast(BinaryIO, f))

    async def __aenter__(self) -> 'LocalMultiPartCreate':
        return self

    async def __aexit__(self,
                        exc_type: Optional[Type[BaseException]],
                        exc_val: Optional[BaseException],
                        exc_tb: Optional[TracebackType]) -> None:
        if exc_val:
            try:
                await self._fs.remove(self._path)
            except FileNotFoundError:
                pass


class LocalAsyncFS(AsyncFS):
    def __init__(self, thread_pool: ThreadPoolExecutor, max_workers=None):
        if not thread_pool:
            thread_pool = ThreadPoolExecutor(max_workers=max_workers)
        self._thread_pool = thread_pool

    def schemes(self) -> Set[str]:
        return {'file'}

    @staticmethod
    def _get_path(url):
        parsed = urllib.parse.urlparse(url)
        if parsed.scheme and parsed.scheme != 'file':
            raise ValueError(f"invalid scheme, expected file: {parsed.scheme}")
        return parsed.path

    async def open(self, url: str) -> ReadableStream:
        f = await blocking_to_async(self._thread_pool, open, self._get_path(url), 'rb')
        return blocking_readable_stream_to_async(self._thread_pool, cast(BinaryIO, f))

    async def open_from(self, url: str, start: int) -> ReadableStream:
        f = await blocking_to_async(self._thread_pool, open, self._get_path(url), 'rb')
        f.seek(start, io.SEEK_SET)
        return blocking_readable_stream_to_async(self._thread_pool, cast(BinaryIO, f))

    async def create(self, url: str, *, retry_writes: bool = True) -> WritableStream:  # pylint: disable=unused-argument
        f = await blocking_to_async(self._thread_pool, open, self._get_path(url), 'wb')
        return blocking_writable_stream_to_async(self._thread_pool, cast(BinaryIO, f))

    async def multi_part_create(
            self,
            sema: asyncio.Semaphore,  # pylint: disable=unused-argument
            url: str,
            num_parts: int) -> MultiPartCreate:
        # create an empty file
        # will be opened r+b to write the parts
        async with await self.create(url):
            pass
        return LocalMultiPartCreate(self, self._get_path(url), num_parts)

    async def statfile(self, url: str) -> LocalStatFileStatus:
        path = self._get_path(url)
        stat_result = await blocking_to_async(self._thread_pool, os.stat, path)
        if stat.S_ISDIR(stat_result.st_mode):
            raise FileNotFoundError(f'is directory: {url}')
        return LocalStatFileStatus(stat_result)

    # entries has no type hint because the return type of os.scandir
    # appears to be a private type, posix.ScandirIterator.
    # >>> import os
    # >>> entries = os.scandir('.')
    # >>> type(entries)
    # <class 'posix.ScandirIterator'>
    # >>> import posix
    # >>> posix.ScandirIterator
    # Traceback (most recent call last):
    #   File "<stdin>", line 1, in <module>
    # AttributeError: module 'posix' has no attribute 'ScandirIterator'
    async def _listfiles_recursive(self, url: str, entries) -> AsyncIterator[FileListEntry]:
        async for file in self._listfiles_flat(url, entries):
            if await file.is_file():
                yield file
            else:
                new_url = await file.url()
                new_path = self._get_path(new_url)
                new_entries = await blocking_to_async(self._thread_pool, os.scandir, new_path)
                async for subfile in self._listfiles_recursive(new_url, new_entries):
                    yield subfile

    async def _listfiles_flat(self, url: str, entries) -> AsyncIterator[FileListEntry]:
        with entries:
            for entry in entries:
                yield LocalFileListEntry(self._thread_pool, url, entry)

    async def listfiles(self, url: str, recursive: bool = False) -> AsyncIterator[FileListEntry]:
        path = self._get_path(url)
        entries = await blocking_to_async(self._thread_pool, os.scandir, path)
        if recursive:
            return self._listfiles_recursive(url, entries)
        return self._listfiles_flat(url, entries)

    async def staturl(self, url: str) -> str:
        path = self._get_path(url)
        stat_result = await blocking_to_async(self._thread_pool, os.stat, path)
        if stat.S_ISDIR(stat_result.st_mode):
            return AsyncFS.DIR
        return AsyncFS.FILE

    async def mkdir(self, url: str) -> None:
        path = self._get_path(url)
        await blocking_to_async(self._thread_pool, os.mkdir, path)

    async def makedirs(self, url: str, exist_ok: bool = False) -> None:
        path = self._get_path(url)
        await blocking_to_async(self._thread_pool, os.makedirs, path, exist_ok=exist_ok)

    async def isfile(self, url: str) -> bool:
        path = self._get_path(url)
        return await blocking_to_async(self._thread_pool, os.path.isfile, path)

    async def isdir(self, url: str) -> bool:
        path = self._get_path(url)
        return await blocking_to_async(self._thread_pool, os.path.isdir, path)

    async def remove(self, url: str) -> None:
        path = self._get_path(url)
        return os.remove(path)

    async def rmtree(self, sema: asyncio.Semaphore, url: str) -> None:
        path = self._get_path(url)
        await blocking_to_async(self._thread_pool, shutil.rmtree, path)


class FileAndDirectoryError(Exception):
    pass


class Transfer:
    DEST_DIR = 'dest_dir'
    DEST_IS_TARGET = 'dest_is_target'
    INFER_DEST = 'infer_dest'

    def __init__(self, src: Union[str, List[str]], dest: str, *, treat_dest_as: str = INFER_DEST):
        if treat_dest_as not in (Transfer.DEST_DIR, Transfer.DEST_IS_TARGET, Transfer.INFER_DEST):
            raise ValueError(f'treat_dest_as invalid: {treat_dest_as}')

        if treat_dest_as == Transfer.DEST_IS_TARGET and isinstance(src, list):
            raise NotADirectoryError(dest)
        if (treat_dest_as == Transfer.INFER_DEST
                and dest.endswith('/')):
            treat_dest_as = Transfer.DEST_DIR

        self.src = src
        self.dest = dest
        self.treat_dest_as = treat_dest_as


class SourceReport:
    def __init__(self, source):
        self._source = source
        self._source_type: Optional[str] = None
        self._files = 0
        self._bytes = 0
        self._errors = 0
        self._complete = 0
        self._first_file_error: Optional[Dict[str, Any]] = None
        self._exception: Optional[Exception] = None

    def set_exception(self, exception: Exception):
        assert not self._exception
        self._exception = exception

    def set_file_error(self, srcfile: str, destfile: str, exception: Exception):
        if self._first_file_error is None:
            self._first_file_error = {
                'srcfile': srcfile,
                'destfile': destfile,
                'exception': exception
            }


class TransferReport:
    _source_report: Union[SourceReport, List[SourceReport]]

    def __init__(self, transfer: Transfer):
        self._transfer = transfer
        if isinstance(transfer.src, str):
            self._source_report = SourceReport(transfer.src)
        else:
            self._source_report = [SourceReport(s) for s in transfer.src]
        self._exception: Optional[Exception] = None

    def set_exception(self, exception: Exception):
        assert not self._exception
        self._exception = exception


class CopyReport:
    def __init__(self, transfer: Union[Transfer, List[Transfer]]):
        self._start_time = time_msecs()
        self._end_time = None
        self._duration = None
        if isinstance(transfer, Transfer):
            self._transfer_report: Union[TransferReport, List[TransferReport]] = TransferReport(transfer)
        else:
            self._transfer_report = [TransferReport(t) for t in transfer]
        self._exception: Optional[Exception] = None

    def set_exception(self, exception: Exception):
        assert not self._exception
        self._exception = exception

    def mark_done(self):
        self._end_time = time_msecs()
        self._duration = self._end_time - self._start_time

    def summarize(self):
        source_reports = []

        def add_source_reports(transfer_report):
            if isinstance(transfer_report._source_report, SourceReport):
                source_reports.append(transfer_report._source_report)
            else:
                source_reports.extend(transfer_report._source_report)

        if isinstance(self._transfer_report, Transfer):
            total_transfers = 1
            add_source_reports(self._transfer_report)
        else:
            total_transfers = len(self._transfer_report)
            for t in self._transfer_report:
                add_source_reports(t)

        total_sources = len(source_reports)
        total_files = sum([sr._files for sr in source_reports])
        total_bytes = sum([sr._bytes for sr in source_reports])

        print('Transfer summary:')
        print(f'  Transfers: {total_transfers}')
        print(f'  Sources: {total_sources}')
        print(f'  Files: {total_files}')
        print(f'  Bytes: {humanize.naturalsize(total_bytes)}')
        print(f'  Time: {humanize_timedelta_msecs(self._duration)}')
        print(f'  Average transfer rate: {humanize.naturalsize(total_bytes / (self._duration / 1000))}/s')

        print('Sources:')
        for sr in source_reports:
            print(f'  {sr._source}: {sr._files} files, {humanize.naturalsize(sr._bytes)}')


class UnexpectedEOFError(Exception):
    pass


class SourceCopier:
    '''This class implements copy from a single source.  In general, a
    transfer will have multiple sources, and a SourceCopier will be
    created for each source.
    '''

    PART_SIZE = 128 * 1024 * 1024

    def __init__(self, router_fs: 'RouterAsyncFS', src: str, dest: str, treat_dest_as: str, dest_type_task):
        self.router_fs = router_fs
        self.src = src
        self.dest = dest
        self.treat_dest_as = treat_dest_as
        self.dest_type_task = dest_type_task

        self.src_is_file: Optional[bool] = None
        self.src_is_dir: Optional[bool] = None

        self.pending = 2
        self.barrier = asyncio.Event()

    async def release_barrier(self):
        self.pending -= 1
        if self.pending == 0:
            self.barrier.set()

    async def _copy_file(self, srcfile: str, destfile: str) -> None:
        assert not destfile.endswith('/')

        async with await self.router_fs.open(srcfile) as srcf:
            try:
                destf = await self.router_fs.create(destfile, retry_writes=False)
            except FileNotFoundError:
                await self.router_fs.makedirs(os.path.dirname(destfile), exist_ok=True)
                destf = await self.router_fs.create(destfile)

            async with destf:
                while True:
                    b = await srcf.read(Copier.BUFFER_SIZE)
                    if not b:
                        return
                    written = await destf.write(b)
                    assert written == len(b)

    async def _copy_part(self, source_report, srcfile, part_number, part_creator, return_exceptions):
        try:
            async with await self.router_fs.open_from(srcfile, part_number * self.PART_SIZE) as srcf:
                async with await part_creator.create_part(part_number, part_number * self.PART_SIZE, retry_writes=False) as destf:
                    n = self.PART_SIZE
                    while n > 0:
                        b = await srcf.read(min(Copier.BUFFER_SIZE, n))
                        # FIXME check expected bytes
                        if not b:
                            return
                        written = await destf.write(b)
                        assert written == len(b)
                        n -= len(b)
        except Exception as e:
            if return_exceptions:
                source_report.set_exception(e)
            else:
                raise

    async def _copy_file_multi_part_main(
            self,
            sema: asyncio.Semaphore,
            source_report: SourceReport,
            srcfile: str,
            srcstat: FileStatus,
            destfile: str,
            return_exceptions: bool):
        size = await srcstat.size()
        if size <= self.PART_SIZE:
            await retry_transient_errors(self._copy_file, srcfile, destfile)
            return

        n_parts = int((size + self.PART_SIZE - 1) / self.PART_SIZE)

        try:
            part_creator = await self.router_fs.multi_part_create(sema, destfile, n_parts)
        except FileNotFoundError:
            await self.router_fs.makedirs(os.path.dirname(destfile), exist_ok=True)
            part_creator = await self.router_fs.multi_part_create(sema, destfile, n_parts)

        async with part_creator:
            await bounded_gather2(sema, *[
                retry_transient_errors(self._copy_part, source_report, srcfile, i, part_creator, return_exceptions)
                for i in range(n_parts)
            ], cancel_on_error=True)

    async def _copy_file_multi_part(
            self,
            sema: asyncio.Semaphore,
            source_report: SourceReport,
            srcfile: str,
            srcstat: FileStatus,
            destfile: str,
            return_exceptions: bool):
        source_report._files += 1
        source_report._bytes += await srcstat.size()
        success = False
        try:
            await self._copy_file_multi_part_main(sema, source_report, srcfile, srcstat, destfile, return_exceptions)
            source_report._complete += 1
            success = True
        except Exception as e:
            if return_exceptions:
                source_report.set_file_error(srcfile, destfile, e)
            else:
                raise e
        finally:
            if not success:
                source_report._errors += 1

    async def _full_dest(self):
        if self.dest_type_task:
            dest_type = await self.dest_type_task
        else:
            dest_type = None

        if (self.treat_dest_as == Transfer.DEST_DIR
                or (self.treat_dest_as == Transfer.INFER_DEST
                    and dest_type == AsyncFS.DIR)):
            # We know dest is a dir, but we're copying to
            # dest/basename(src), and we don't know its type.
            return url_join(self.dest, url_basename(self.src.rstrip('/'))), None

        if (self.treat_dest_as == Transfer.DEST_IS_TARGET
                and self.dest.endswith('/')):
            dest_type = AsyncFS.DIR

        return self.dest, dest_type

    async def copy_as_file(self,
                           sema: asyncio.Semaphore,  # pylint: disable=unused-argument
                           source_report: SourceReport,
                           return_exceptions: bool):
        try:
            src = self.src
            if src.endswith('/'):
                return
            try:
                srcstat = await self.router_fs.statfile(src)
            except FileNotFoundError:
                self.src_is_file = False
                return
            self.src_is_file = True
        finally:
            await self.release_barrier()

        await self.barrier.wait()

        if self.src_is_dir:
            raise FileAndDirectoryError(self.src)

        source_report._source_type = AsyncFS.FILE

        full_dest, full_dest_type = await self._full_dest()
        if full_dest_type == AsyncFS.DIR:
            raise IsADirectoryError(full_dest)

        await self._copy_file_multi_part(sema, source_report, src, srcstat, full_dest, return_exceptions)

    async def copy_as_dir(self, sema: asyncio.Semaphore, source_report: SourceReport, return_exceptions: bool):
        try:
            src = self.src
            if not src.endswith('/'):
                src = src + '/'
            try:
                srcentries = await self.router_fs.listfiles(src, recursive=True)
            except (NotADirectoryError, FileNotFoundError):
                self.src_is_dir = False
                return
            self.src_is_dir = True
        finally:
            await self.release_barrier()

        await self.barrier.wait()

        if self.src_is_file:
            raise FileAndDirectoryError(self.src)

        source_report._source_type = AsyncFS.DIR

        full_dest, full_dest_type = await self._full_dest()
        if full_dest_type == AsyncFS.FILE:
            raise NotADirectoryError(full_dest)

        async def copy_source(srcentry):
            srcfile = srcentry.url_maybe_trailing_slash()
            assert srcfile.startswith(src)

            # skip files with empty names
            if srcfile.endswith('/'):
                return

            relsrcfile = srcfile[len(src):]
            assert not relsrcfile.startswith('/')

            await self._copy_file_multi_part(sema, source_report, srcfile, await srcentry.status(), url_join(full_dest, relsrcfile), return_exceptions)

        await bounded_gather2(sema, *[
            copy_source(srcentry)
            async for srcentry in srcentries], cancel_on_error=True)

    async def copy(self, sema: asyncio.Semaphore, source_report: SourceReport, return_exceptions: bool):
        try:
            # gather with return_exceptions=True to make copy
            # deterministic with respect to exceptions
            results = await asyncio.gather(
                self.copy_as_file(sema, source_report, return_exceptions), self.copy_as_dir(sema, source_report, return_exceptions),
                return_exceptions=True)

            assert self.pending == 0

            for result in results:
                if isinstance(result, Exception):
                    raise result

            assert (self.src_is_file is None) == self.src.endswith('/')
            assert self.src_is_dir is not None
            if (self.src_is_file is False or self.src.endswith('/')) and not self.src_is_dir:
                raise FileNotFoundError(self.src)

        except Exception as e:
            if return_exceptions:
                source_report.set_exception(e)
            else:
                raise e


class Copier:
    '''
    This class implements copy for a list of transfers.
    '''

    BUFFER_SIZE = 256 * 1024

    def __init__(self, router_fs):
        self.router_fs = router_fs

    async def _dest_type(self, transfer: Transfer):
        '''Return the (real or assumed) type of `dest`.

        If the transfer assumes the type of `dest`, return that rather
        than the real type.  A return value of `None` mean `dest` does
        not exist.
        '''
        assert transfer.treat_dest_as != Transfer.DEST_IS_TARGET

        if (transfer.treat_dest_as == Transfer.DEST_DIR
                or isinstance(transfer.src, list)
                or transfer.dest.endswith('/')):
            return AsyncFS.DIR

        assert not transfer.dest.endswith('/')
        try:
            dest_type = await self.router_fs.staturl(transfer.dest)
        except FileNotFoundError:
            dest_type = None

        return dest_type

    async def copy_source(self, sema: asyncio.Semaphore, transfer: Transfer, source_report: SourceReport, src: str, dest_type_task, return_exceptions: bool):
        src_copier = SourceCopier(self.router_fs, src, transfer.dest, transfer.treat_dest_as, dest_type_task)
        await src_copier.copy(sema, source_report, return_exceptions)

    async def _copy_one_transfer(self, sema: asyncio.Semaphore, transfer_report: TransferReport, transfer: Transfer, return_exceptions: bool):
        try:
            if transfer.treat_dest_as == Transfer.INFER_DEST:
                dest_type_task: Optional[asyncio.Task] = asyncio.create_task(self._dest_type(transfer))
            else:
                dest_type_task = None

            try:
                src = transfer.src
                src_report = transfer_report._source_report
                if isinstance(src, str):
                    assert isinstance(src_report, SourceReport)
                    await self.copy_source(sema, transfer, src_report, src, dest_type_task, return_exceptions)
                else:
                    assert isinstance(src_report, list)
                    if transfer.treat_dest_as == Transfer.DEST_IS_TARGET:
                        raise NotADirectoryError(transfer.dest)

                    await bounded_gather2(sema, *[
                        self.copy_source(sema, transfer, r, s, dest_type_task, return_exceptions)
                        for r, s in zip(src_report, src)
                    ], cancel_on_error=True)

                # raise potential exception
                if dest_type_task:
                    await dest_type_task
            finally:
                if dest_type_task:
                    await asyncio.wait([dest_type_task])
        except Exception as e:
            if return_exceptions:
                transfer_report.set_exception(e)
            else:
                raise e

    async def copy(self, sema: asyncio.Semaphore, copy_report: CopyReport, transfer: Union[Transfer, List[Transfer]], return_exceptions: bool):
        transfer_report = copy_report._transfer_report
        try:
            if isinstance(transfer, Transfer):
                assert isinstance(transfer_report, TransferReport)
                await self._copy_one_transfer(sema, transfer_report, transfer, return_exceptions)
                return

            assert isinstance(transfer_report, list)
            await bounded_gather2(sema, *[
                self._copy_one_transfer(sema, r, t, return_exceptions)
                for r, t in zip(transfer_report, transfer)
            ], return_exceptions=return_exceptions, cancel_on_error=True)
        except Exception as e:
            if return_exceptions:
                copy_report.set_exception(e)
            else:
                raise e


class RouterAsyncFS(AsyncFS):
    def __init__(self, default_scheme: Optional[str], filesystems: List[AsyncFS]):
        scheme_fs = {}
        schemes = set()
        for fs in filesystems:
            for scheme in fs.schemes():
                if scheme not in schemes:
                    scheme_fs[scheme] = fs
                    schemes.add(scheme)

        if default_scheme not in schemes:
            raise ValueError(f'default scheme {default_scheme} not in set of schemes: {", ".join(schemes)}')

        self._default_scheme = default_scheme
        self._filesystems = filesystems
        self._schemes = schemes
        self._scheme_fs = scheme_fs

    def schemes(self) -> Set[str]:
        return self._schemes

    def _get_fs(self, url):
        parsed = urllib.parse.urlparse(url)
        if not parsed.scheme:
            if self._default_scheme:
                parsed = parsed._replace(scheme=self._default_scheme)
                url = urllib.parse.urlunparse(parsed)
            else:
                raise ValueError(f"no default scheme and URL has no scheme: {url}")

        fs = self._scheme_fs.get(parsed.scheme)
        if fs is None:
            raise ValueError(f"unknown scheme: {parsed.scheme}")

        return fs

    async def open(self, url: str) -> ReadableStream:
        fs = self._get_fs(url)
        return await fs.open(url)

    async def open_from(self, url: str, start: int) -> ReadableStream:
        fs = self._get_fs(url)
        return await fs.open_from(url, start)

    async def create(self, url: str, *, retry_writes: bool = True) -> WritableStream:
        fs = self._get_fs(url)
        return await fs.create(url, retry_writes=retry_writes)

    async def multi_part_create(
            self,
            sema: asyncio.Semaphore,
            url: str,
            num_parts: int) -> MultiPartCreate:
        fs = self._get_fs(url)
        return await fs.multi_part_create(sema, url, num_parts)

    async def statfile(self, url: str) -> FileStatus:
        fs = self._get_fs(url)
        return await fs.statfile(url)

    async def listfiles(self, url: str, recursive: bool = False) -> AsyncIterator[FileListEntry]:
        fs = self._get_fs(url)
        return await fs.listfiles(url, recursive)

    async def staturl(self, url: str) -> str:
        fs = self._get_fs(url)
        return await fs.staturl(url)

    async def mkdir(self, url: str) -> None:
        fs = self._get_fs(url)
        return await fs.mkdir(url)

    async def makedirs(self, url: str, exist_ok: bool = False) -> None:
        fs = self._get_fs(url)
        return await fs.makedirs(url, exist_ok=exist_ok)

    async def isfile(self, url: str) -> bool:
        fs = self._get_fs(url)
        return await fs.isfile(url)

    async def isdir(self, url: str) -> bool:
        fs = self._get_fs(url)
        return await fs.isdir(url)

    async def remove(self, url: str) -> None:
        fs = self._get_fs(url)
        return await fs.remove(url)

    async def rmtree(self, sema: asyncio.Semaphore, url: str) -> None:
        fs = self._get_fs(url)
        return await fs.rmtree(sema, url)

    async def close(self) -> None:
        for fs in self._filesystems:
            await fs.close()

    async def copy(self, sema: asyncio.Semaphore, transfer: Union[Transfer, List[Transfer]], return_exceptions: bool = False) -> CopyReport:
        copier = Copier(self)
        copy_report = CopyReport(transfer)
        await copier.copy(sema, copy_report, transfer, return_exceptions)
        copy_report.mark_done()
        return copy_report
