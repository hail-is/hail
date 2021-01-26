from typing import TypeVar, Any, Optional, List, Type, BinaryIO, cast, Set, AsyncIterator, Union, Dict
from types import TracebackType
import abc
import os
import os.path
import stat
import shutil
import asyncio
from concurrent.futures import ThreadPoolExecutor
import urllib.parse
from hailtop.utils import blocking_to_async, url_basename, url_join, AsyncWorkerPool, WaitableSharedPool
from .stream import ReadableStream, WritableStream, blocking_readable_stream_to_async, blocking_writable_stream_to_async

AsyncFSType = TypeVar('AsyncFSType', bound='AsyncFS')


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
    async def create_part(self, number: int, start: int):
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
    async def create(self, url: str) -> WritableStream:
        pass

    @abc.abstractmethod
    async def multi_part_create(self, url: str, num_parts: int) -> MultiPartCreate:
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
    async def rmtree(self, url: str) -> None:
        pass

    async def touch(self, url: str) -> None:
        async with await self.create(url):
            pass

    async def close(self) -> None:
        pass

    async def __aenter__(self) -> AsyncFSType:
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
    def __init__(self, fs: AsyncFS, path: str, num_parts: int):
        self._fs = fs
        self._path = path
        self._num_parts = num_parts

    async def create_part(self, number: int, start: int):
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

    async def create(self, url: str) -> WritableStream:
        f = await blocking_to_async(self._thread_pool, open, self._get_path(url), 'wb')
        return blocking_writable_stream_to_async(self._thread_pool, cast(BinaryIO, f))

    async def multi_part_create(self, url: str, num_parts: int) -> MultiPartCreate:
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

    async def rmtree(self, url: str) -> None:
        path = self._get_path(url)
        await blocking_to_async(self._thread_pool, shutil.rmtree, path)


class FileAndDirectoryError(Exception):
    pass


class Transfer:
    TARGET_DIR = 'target_dir'
    TARGET_FILE = 'target_file'
    INFER_TARGET = 'infer_target'

    def __init__(self, src: Union[str, List[str]], dest: str, *, treat_dest_as: str = INFER_TARGET):
        if treat_dest_as not in (Transfer.TARGET_DIR, Transfer.TARGET_FILE, Transfer.INFER_TARGET):
            raise ValueError(f'treat_dest_as invalid: {treat_dest_as}')

        if treat_dest_as == Transfer.TARGET_FILE and isinstance(src, list):
            raise NotADirectoryError(dest)

        self.src = src
        self.dest = dest
        self.treat_dest_as = treat_dest_as


class SourceReport:
    def __init__(self, source):
        self._source = source
        self._source_type: Optional[str] = None
        self._files = 0
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

    def raise_first_exception(self):
        if self._exception:
            raise self._exception
        if self._first_file_error:
            raise self._first_file_error['exception']


class TransferReport:
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

    def raise_first_exception(self):
        if self._exception:
            raise self._exception
        if isinstance(self._source_report, SourceReport):
            self._source_report.raise_first_exception()
        else:
            for s in self._source_report:
                s.raise_first_exception()


class CopyReport:
    def __init__(self, transfer: Union[Transfer, List[Transfer]]):
        if isinstance(transfer, Transfer):
            self._transfer_report = TransferReport(transfer)
        else:
            self._transfer_report = [TransferReport(t) for t in transfer]
        self._exception: Optional[Exception] = None

    def set_exception(self, exception: Exception):
        assert not self._exception
        self._exception = exception

    def raise_first_exception(self):
        if self._exception:
            raise self._exception
        if isinstance(self._transfer_report, TransferReport):
            self._transfer_report.raise_first_exception()
        else:
            for t in self._transfer_report:
                t.raise_first_exception()


class SourceCopier:
    '''This class implements copy from a single source.  In general, a
    transfer will have multiple sources, and a SourceCopier will be
    created for each source.
    '''

    def __init__(self, router_fs: 'RouterAsyncFS', src: str, dest: str, treat_dest_as: str, dest_type_task):
        self.router_fs = router_fs
        self.src = src
        self.dest = dest
        self.treat_dest_as = treat_dest_as
        self.dest_type_task = dest_type_task

        self.src_is_file = None
        self.src_is_dir = None

        self.pending = 2
        self.barrier = asyncio.Event()

    async def release_barrier(self):
        self.pending -= 1
        if self.pending == 0:
            self.barrier.set()

    async def release_barrier_and_wait(self):
        await self.release_barrier()
        await self.barrier.wait()

    async def _copy_file(self, source_report: SourceReport, srcfile: str, destfile: str) -> None:
        source_report._files += 1
        success = False
        try:
            assert not destfile.endswith('/')

            async with await self.router_fs.open(srcfile) as srcf:
                try:
                    destf = await self.router_fs.create(destfile)
                except FileNotFoundError:
                    await self.router_fs.makedirs(os.path.dirname(destfile), exist_ok=True)
                    destf = await self.router_fs.create(destfile)

                async with destf:
                    while True:
                        b = await srcf.read(Copier.BUFFER_SIZE)
                        if not b:
                            return
                        await destf.write(b)
            source_report._complete += 1
            success = True
        except Exception as e:
            source_report.set_file_error(srcfile, destfile, e)
        finally:
            if not success:
                source_report._errors += 1

    async def _full_dest(self):
        dest_type = await self.dest_type_task

        if (self.treat_dest_as == Transfer.TARGET_DIR
                or self.dest.endswith('/')
                or (self.treat_dest_as == Transfer.INFER_TARGET
                    and dest_type == AsyncFS.DIR)):
            if dest_type is None:
                raise FileNotFoundError(self.dest)
            if dest_type == AsyncFS.FILE:
                raise NotADirectoryError(self.dest)
            assert dest_type == AsyncFS.DIR
            # We know dest is a dir, but we're copying to
            # dest/basename(src), and we don't know its type.
            return url_join(self.dest, url_basename(self.src.rstrip('/'))), None

        assert not self.dest.endswith('/')
        return self.dest, dest_type

    async def copy_as_file(self,
                           worker_pool: AsyncWorkerPool,  # pylint: disable=unused-argument
                           source_report: SourceReport):
        src = self.src
        if src.endswith('/'):
            await self.release_barrier()
            return

        try:
            # currently unused; size will be use to do mutli-part
            # uploads
            await self.router_fs.statfile(src)
        except FileNotFoundError:
            self.src_is_file = False
            await self.release_barrier()
            return

        self.src_is_file = True
        await self.release_barrier_and_wait()

        if self.src_is_dir:
            raise FileAndDirectoryError(self.src)

        source_report._source_type = AsyncFS.FILE

        full_dest, full_dest_type = await self._full_dest()
        if full_dest_type == AsyncFS.DIR:
            raise IsADirectoryError(full_dest)

        await self._copy_file(source_report, src, full_dest)

    async def copy_as_dir(self, worker_pool: AsyncWorkerPool, source_report: SourceReport):
        src = self.src
        if not src.endswith('/'):
            src = src + '/'

        try:
            srcentries = await self.router_fs.listfiles(src, recursive=True)
        except (NotADirectoryError, FileNotFoundError):
            self.src_is_dir = False
            await self.release_barrier()
            return

        self.src_is_dir = True
        await self.release_barrier_and_wait()

        if self.src_is_file:
            raise FileAndDirectoryError(self.src)

        source_report._source_type = AsyncFS.DIR

        full_dest, full_dest_type = await self._full_dest()
        if full_dest_type == AsyncFS.FILE:
            raise NotADirectoryError(full_dest)

        async with WaitableSharedPool(worker_pool) as pool:
            async for srcentry in srcentries:
                srcfile = srcentry.url_maybe_trailing_slash()
                assert srcfile.startswith(src)

                # skip files with empty names
                if srcfile.endswith('/'):
                    continue

                relsrcfile = srcfile[len(src):]
                assert not relsrcfile.startswith('/')

                await pool.call(self._copy_file, source_report, srcfile, url_join(full_dest, relsrcfile))

    async def copy(self, worker_pool: AsyncWorkerPool, source_report: SourceReport):
        try:
            # gather with return_exceptions=True to make copy
            # deterministic with respect to exceptions
            results = await asyncio.gather(
                self.copy_as_file(worker_pool, source_report), self.copy_as_dir(worker_pool, source_report),
                return_exceptions=True)

            assert self.pending == 0
            assert (self.src_is_file is None) == self.src.endswith('/')
            assert self.src_is_dir is not None

            if (self.src_is_file is False or self.src.endswith('/')) and not self.src_is_dir:
                raise FileNotFoundError(self.src)
            for result in results:
                if isinstance(result, Exception):
                    raise result
        except Exception as e:
            source_report.set_exception(e)


class Copier:
    '''
    This class implements copy for a list of transfers.
    '''

    BUFFER_SIZE = 8192

    def __init__(self, router_fs):
        self.router_fs = router_fs

    async def _dest_type(self, transfer: Transfer):
        '''Return the (real or assumed) type of `dest`.

        If the transfer assumes the type of `dest`, return that rather
        than the real type.  A return value of `None` mean `dest` does
        not exist.
        '''
        if (transfer.treat_dest_as == Transfer.TARGET_DIR
                or isinstance(transfer.src, list)
                or transfer.dest.endswith('/')):
            return AsyncFS.DIR

        if transfer.treat_dest_as == Transfer.TARGET_FILE:
            return AsyncFS.FILE

        assert not transfer.dest.endswith('/')
        try:
            dest_type = await self.router_fs.staturl(transfer.dest)
        except FileNotFoundError:
            dest_type = None

        return dest_type

    async def copy_source(self, worker_pool: AsyncWorkerPool, transfer: Transfer, source_report: SourceReport, src, dest_type_task):
        src_copier = SourceCopier(self.router_fs, src, transfer.dest, transfer.treat_dest_as, dest_type_task)
        await src_copier.copy(worker_pool, source_report)

    async def _copy_one_transfer(self, worker_pool: AsyncWorkerPool, transfer_report: TransferReport, transfer: Transfer):
        try:
            dest_type_task = asyncio.create_task(self._dest_type(transfer))
            dest_type_task_awaited = False

            try:
                src = transfer.src
                if isinstance(src, str):
                    await self.copy_source(worker_pool, transfer, transfer_report._source_report, src, dest_type_task)
                else:
                    if transfer.treat_dest_as == Transfer.TARGET_FILE:
                        raise NotADirectoryError(transfer.dest)

                    async with WaitableSharedPool(worker_pool) as pool:
                        for r, s in zip(transfer_report._source_report, src):
                            await pool.call(self.copy_source, worker_pool, transfer, r, s, dest_type_task)

                # raise potential exception
                dest_type_task_awaited = True
                await dest_type_task
            finally:
                if not dest_type_task_awaited:
                    # retrieve dest_type_task exception to avoid
                    # "Task exception was never retrieved" errors
                    try:
                        dest_type_task_awaited = True
                        await dest_type_task
                    except:
                        pass
        except Exception as e:
            transfer_report.set_exception(e)

    async def copy(self, worker_pool: AsyncWorkerPool, copy_report: CopyReport, transfer: Union[Transfer, List[Transfer]]):
        try:
            if isinstance(transfer, Transfer):
                await self._copy_one_transfer(worker_pool, copy_report._transfer_report, transfer)
                return

            async with WaitableSharedPool(worker_pool) as pool:
                for r, t in zip(copy_report._transfer_report, transfer):
                    await pool.call(self._copy_one_transfer, worker_pool, r, t)
        except Exception as e:
            copy_report.set_exception(e)


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

    async def create(self, url: str) -> WritableStream:
        fs = self._get_fs(url)
        return await fs.create(url)

    async def multi_part_create(self, url: str, num_parts: int) -> MultiPartCreate:
        fs = self._get_fs(url)
        return await fs.multi_part_create(url, num_parts)

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

    async def rmtree(self, url: str) -> None:
        fs = self._get_fs(url)
        return await fs.rmtree(url)

    async def close(self) -> None:
        for fs in self._filesystems:
            await fs.close()

    async def copy(self, transfer: Union[Transfer, List[Transfer]], raise_first_exception=True):
        worker_pool = AsyncWorkerPool(50)
        try:
            copier = Copier(self)
            copy_report = CopyReport(transfer)
            await copier.copy(worker_pool, copy_report, transfer)
            if raise_first_exception:
                copy_report.raise_first_exception()
            return copy_report
        finally:
            worker_pool.shutdown()
