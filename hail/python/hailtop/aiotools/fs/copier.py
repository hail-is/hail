from typing import Any, AsyncIterator, Awaitable, Optional, List, Union, Dict, Callable, Tuple
import os
import os.path
import asyncio
import functools
import humanize


from ...utils import (
    retry_transient_errors,
    url_basename,
    url_join,
    bounded_gather2,
    time_msecs,
    humanize_timedelta_msecs,
)
from ..weighted_semaphore import WeightedSemaphore
from .exceptions import FileAndDirectoryError, UnexpectedEOFError
from .fs import MultiPartCreate, FileStatus, AsyncFS, FileListEntry


class Transfer:
    DEST_DIR = 'dest_dir'
    DEST_IS_TARGET = 'dest_is_target'
    INFER_DEST = 'infer_dest'

    def __init__(self, src: Union[str, List[str]], dest: str, *, treat_dest_as: str = INFER_DEST):
        if treat_dest_as not in (Transfer.DEST_DIR, Transfer.DEST_IS_TARGET, Transfer.INFER_DEST):
            raise ValueError(f'treat_dest_as invalid: {treat_dest_as}')

        if treat_dest_as == Transfer.DEST_IS_TARGET and isinstance(src, list):
            raise NotADirectoryError(dest)
        if treat_dest_as == Transfer.INFER_DEST and dest.endswith('/'):
            treat_dest_as = Transfer.DEST_DIR

        self.src = src
        self.dest = dest
        self.treat_dest_as = treat_dest_as


class SourceReport:
    def __init__(
        self,
        source,
        *,
        files_listener: Optional[Callable[[int], None]] = None,
        bytes_listener: Optional[Callable[[int], None]] = None,
    ):
        self._source = source
        self._files_listener = files_listener
        self._bytes_listener = bytes_listener
        self._source_type: Optional[str] = None
        self._files = 0
        self._bytes = 0
        self._errors = 0
        self._complete = 0
        self._first_file_error: Optional[Dict[str, Any]] = None
        self._exception: Optional[Exception] = None

    def start_files(self, n_files: int):
        self._files += n_files
        if self._files_listener:
            self._files_listener(n_files)

    def start_bytes(self, n_bytes: int):
        self._bytes += n_bytes
        if self._bytes_listener:
            self._bytes_listener(n_bytes)

    def finish_files(self, n_files: int, failed: bool = False):
        if failed:
            self._errors += n_files
        else:
            self._complete += n_files
        if self._files_listener:
            self._files_listener(-n_files)

    def finish_bytes(self, n_bytes: int):
        if self._bytes_listener:
            self._bytes_listener(-n_bytes)

    def set_exception(self, exception: Exception):
        assert not self._exception
        self._exception = exception

    def set_file_error(self, srcfile: str, destfile: str, exception: Exception):
        if self._first_file_error is None:
            self._first_file_error = {'srcfile': srcfile, 'destfile': destfile, 'exception': exception}


class TransferReport:
    _source_report: Union[SourceReport, List[SourceReport]]

    def __init__(
        self,
        transfer: Transfer,
        *,
        files_listener: Optional[Callable[[int], None]] = None,
        bytes_listener: Optional[Callable[[int], None]] = None,
    ):
        self._transfer = transfer
        if isinstance(transfer.src, str):
            self._source_report = SourceReport(
                transfer.src, files_listener=files_listener, bytes_listener=bytes_listener
            )
        else:
            self._source_report = [
                SourceReport(s, files_listener=files_listener, bytes_listener=bytes_listener) for s in transfer.src
            ]
        self._exception: Optional[Exception] = None

    def set_exception(self, exception: Exception):
        assert not self._exception
        self._exception = exception


class CopyReport:
    def __init__(
        self,
        transfer: Union[Transfer, List[Transfer]],
        *,
        files_listener: Optional[Callable[[int], None]] = None,
        bytes_listener: Optional[Callable[[int], None]] = None,
    ):
        self._start_time = time_msecs()
        self._end_time: Optional[int] = None
        self._duration: Optional[int] = None
        if isinstance(transfer, Transfer):
            self._transfer_report: Union[TransferReport, List[TransferReport]] = TransferReport(
                transfer, files_listener=files_listener, bytes_listener=bytes_listener
            )
        else:
            self._transfer_report = [
                TransferReport(t, files_listener=files_listener, bytes_listener=bytes_listener) for t in transfer
            ]
        self._exception: Optional[Exception] = None

    def set_exception(self, exception: Exception):
        assert not self._exception
        self._exception = exception

    def mark_done(self):
        self._end_time = time_msecs()
        self._duration = self._end_time - self._start_time

    def summarize(self, include_sources: bool = True):
        source_reports = []

        def add_source_reports(transfer_report):
            if isinstance(transfer_report._source_report, SourceReport):
                source_reports.append(transfer_report._source_report)
            else:
                source_reports.extend(transfer_report._source_report)

        if isinstance(self._transfer_report, TransferReport):
            total_transfers = 1
            add_source_reports(self._transfer_report)
        else:
            total_transfers = len(self._transfer_report)
            for t in self._transfer_report:
                add_source_reports(t)

        total_sources = len(source_reports)
        total_files = sum(sr._files for sr in source_reports)
        total_bytes = sum(sr._bytes for sr in source_reports)

        print('Transfer summary:')
        print(f'  Transfers: {total_transfers}')
        print(f'  Sources: {total_sources}')
        print(f'  Files: {total_files}')
        print(f'  Bytes: {humanize.naturalsize(total_bytes)}')
        print(f'  Time: {humanize_timedelta_msecs(self._duration)}')
        assert self._duration is not None
        if self._duration > 0:
            bandwidth = humanize.naturalsize(total_bytes / (self._duration / 1000))
            print(f'  Average bandwidth: {bandwidth}/s')
            file_rate = total_files / (self._duration / 1000)
            print(f'  Average file rate: {file_rate:,.1f}/s')

        if include_sources:
            print('Sources:')
            for sr in source_reports:
                print(f'  {sr._source}: {sr._files} files, {humanize.naturalsize(sr._bytes)}')


class SourceCopier:
    """This class implements copy from a single source.  In general, a
    transfer will have multiple sources, and a SourceCopier will be
    created for each source.
    """

    def __init__(
        self, router_fs: AsyncFS, xfer_sema: WeightedSemaphore, src: str, dest: str, treat_dest_as: str, dest_type_task
    ):
        self.router_fs = router_fs
        self.xfer_sema = xfer_sema
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

    async def _copy_file(self, source_report: SourceReport, srcfile: str, size: int, destfile: str) -> None:
        assert not destfile.endswith('/')

        async with self.xfer_sema.acquire_manager(min(Copier.BUFFER_SIZE, size)):
            async with await self.router_fs.open(srcfile) as srcf:
                try:
                    dest_cm = await self.router_fs.create(destfile, retry_writes=False)
                except FileNotFoundError:
                    await self.router_fs.makedirs(os.path.dirname(destfile), exist_ok=True)
                    dest_cm = await self.router_fs.create(destfile)

                async with dest_cm as destf:
                    while True:
                        b = await srcf.read(Copier.BUFFER_SIZE)
                        if not b:
                            return
                        written = await destf.write(b)
                        assert written == len(b)
                        source_report.finish_bytes(written)

    async def _copy_part(
        self,
        source_report: SourceReport,
        part_size: int,
        srcfile: str,
        part_number: int,
        this_part_size: int,
        part_creator: MultiPartCreate,
        return_exceptions: bool,
    ) -> None:
        total_written = 0
        try:
            async with self.xfer_sema.acquire_manager(min(Copier.BUFFER_SIZE, this_part_size)):
                async with await self.router_fs.open_from(
                    srcfile, part_number * part_size, length=this_part_size
                ) as srcf:
                    async with await part_creator.create_part(
                        part_number, part_number * part_size, size_hint=this_part_size
                    ) as destf:
                        n = this_part_size
                        while n > 0:
                            b = await srcf.read(min(Copier.BUFFER_SIZE, n))
                            if len(b) == 0:
                                raise UnexpectedEOFError()
                            written = await destf.write(b)
                            assert written == len(b)
                            total_written += written
                            n -= len(b)
            source_report.finish_bytes(total_written)
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
        return_exceptions: bool,
    ):
        size = await srcstat.size()

        part_size = self.router_fs.copy_part_size(destfile)

        if size <= part_size:
            await retry_transient_errors(self._copy_file, source_report, srcfile, size, destfile)
            return

        n_parts, rem = divmod(size, part_size)
        if rem:
            n_parts += 1

        try:
            part_creator = await self.router_fs.multi_part_create(sema, destfile, n_parts)
        except FileNotFoundError:
            await self.router_fs.makedirs(os.path.dirname(destfile), exist_ok=True)
            part_creator = await self.router_fs.multi_part_create(sema, destfile, n_parts)

        async with part_creator:

            async def f(i):
                this_part_size = rem if i == n_parts - 1 and rem else part_size
                await retry_transient_errors(
                    self._copy_part,
                    source_report,
                    part_size,
                    srcfile,
                    i,
                    this_part_size,
                    part_creator,
                    return_exceptions,
                )

            await bounded_gather2(sema, *[functools.partial(f, i) for i in range(n_parts)], cancel_on_error=True)

    async def _copy_file_multi_part(
        self,
        sema: asyncio.Semaphore,
        source_report: SourceReport,
        srcfile: str,
        srcstat: FileStatus,
        destfile: str,
        return_exceptions: bool,
    ) -> None:
        success = False
        try:
            await self._copy_file_multi_part_main(sema, source_report, srcfile, srcstat, destfile, return_exceptions)
            success = True
        except Exception as e:
            if return_exceptions:
                source_report.set_file_error(srcfile, destfile, e)
            else:
                raise e
        finally:
            source_report.finish_files(1, failed=not success)

    async def _full_dest(self):
        if self.dest_type_task:
            dest_type = await self.dest_type_task
        else:
            dest_type = None

        if self.treat_dest_as == Transfer.DEST_DIR or (
            self.treat_dest_as == Transfer.INFER_DEST and dest_type == AsyncFS.DIR
        ):
            # We know dest is a dir, but we're copying to
            # dest/basename(src), and we don't know its type.
            return url_join(self.dest, url_basename(self.src.rstrip('/'))), None

        if self.treat_dest_as == Transfer.DEST_IS_TARGET and self.dest.endswith('/'):
            dest_type = AsyncFS.DIR

        return self.dest, dest_type

    async def copy_as_file(
        self,
        sema: asyncio.Semaphore,  # pylint: disable=unused-argument
        source_report: SourceReport,
        return_exceptions: bool,
    ):
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

        source_report.start_files(1)
        source_report.start_bytes(await srcstat.size())
        await self._copy_file_multi_part(sema, source_report, src, srcstat, full_dest, return_exceptions)

    async def copy_as_dir(self, sema: asyncio.Semaphore, source_report: SourceReport, return_exceptions: bool):
        async def files_iterator() -> AsyncIterator[FileListEntry]:
            return await self.router_fs.listfiles(src, recursive=True)

        try:
            src = self.src
            if not src.endswith('/'):
                src = src + '/'

            try:
                srcentries: Optional[AsyncIterator[FileListEntry]] = await files_iterator()
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

        async def copy_source(srcentry: FileListEntry) -> None:
            srcfile = await srcentry.url_maybe_trailing_slash()
            assert srcfile.startswith(src)

            # skip files with empty names
            if srcfile.endswith('/'):
                return

            relsrcfile = srcfile[len(src) :]
            assert not relsrcfile.startswith('/')

            await self._copy_file_multi_part(
                sema,
                source_report,
                srcfile,
                await srcentry.status(),
                url_join(full_dest, relsrcfile),
                return_exceptions,
            )

        async def create_copies() -> Tuple[List[Callable[[], Awaitable[None]]], int]:
            nonlocal srcentries
            bytes_to_copy = 0
            if srcentries is None:
                srcentries = await files_iterator()
            try:
                copy_thunks = []
                async for srcentry in srcentries:
                    # In cloud FSes, status and size never make a network request. In local FS, they
                    # can make system calls on symlinks. This line will be fairly expensive if
                    # copying a tree with a lot of symlinks.
                    bytes_to_copy += await (await srcentry.status()).size()
                    copy_thunks.append(functools.partial(copy_source, srcentry))
                return (copy_thunks, bytes_to_copy)
            finally:
                srcentries = None

        copies, bytes_to_copy = await retry_transient_errors(create_copies)
        source_report.start_files(len(copies))
        source_report.start_bytes(bytes_to_copy)
        await bounded_gather2(sema, *copies, cancel_on_error=True)

    async def copy(self, sema: asyncio.Semaphore, source_report: SourceReport, return_exceptions: bool):
        try:
            # gather with return_exceptions=True to make copy
            # deterministic with respect to exceptions
            results = await asyncio.gather(
                self.copy_as_file(sema, source_report, return_exceptions),
                self.copy_as_dir(sema, source_report, return_exceptions),
                return_exceptions=True,
            )

            assert self.pending == 0

            for result in results:
                if isinstance(result, BaseException):
                    raise result

            assert (self.src_is_file is None) == self.src.endswith('/')
            assert self.src_is_dir is not None, repr((
                results,
                self.src_is_file,
                self.src_is_dir,
                self.src,
                self.dest,
                self.barrier,
                self.pending,
            ))
            if (self.src_is_file is False or self.src.endswith('/')) and not self.src_is_dir:
                raise FileNotFoundError(self.src)

        except Exception as e:
            if return_exceptions:
                source_report.set_exception(e)
            else:
                raise e


class Copier:
    """
    This class implements copy for a list of transfers.
    """

    BUFFER_SIZE = 8 * 1024 * 1024

    @staticmethod
    async def copy(
        fs: AsyncFS,
        sema: asyncio.Semaphore,
        transfer: Union[Transfer, List[Transfer]],
        return_exceptions: bool = False,
        *,
        files_listener: Optional[Callable[[int], None]] = None,
        bytes_listener: Optional[Callable[[int], None]] = None,
    ) -> CopyReport:
        copier = Copier(fs)
        copy_report = CopyReport(transfer, files_listener=files_listener, bytes_listener=bytes_listener)
        await copier._copy(sema, copy_report, transfer, return_exceptions)
        copy_report.mark_done()
        return copy_report

    def __init__(self, router_fs):
        self.router_fs = router_fs
        # This is essentially a limit on amount of memory in temporary
        # buffers during copying.  We allow ~10 full-sized copies to
        # run concurrently.
        self.xfer_sema = WeightedSemaphore(100 * Copier.BUFFER_SIZE)

    async def _dest_type(self, transfer: Transfer):
        """Return the (real or assumed) type of `dest`.

        If the transfer assumes the type of `dest`, return that rather
        than the real type.  A return value of `None` mean `dest` does
        not exist.
        """
        assert transfer.treat_dest_as != Transfer.DEST_IS_TARGET

        if transfer.treat_dest_as == Transfer.DEST_DIR or isinstance(transfer.src, list) or transfer.dest.endswith('/'):
            return AsyncFS.DIR

        assert not transfer.dest.endswith('/')
        try:
            dest_type = await self.router_fs.staturl(transfer.dest)
        except FileNotFoundError:
            dest_type = None

        return dest_type

    async def copy_source(
        self,
        sema: asyncio.Semaphore,
        transfer: Transfer,
        source_report: SourceReport,
        src: str,
        dest_type_task,
        return_exceptions: bool,
    ):
        src_copier = SourceCopier(
            self.router_fs, self.xfer_sema, src, transfer.dest, transfer.treat_dest_as, dest_type_task
        )
        await src_copier.copy(sema, source_report, return_exceptions)

    async def _copy_one_transfer(
        self, sema: asyncio.Semaphore, transfer_report: TransferReport, transfer: Transfer, return_exceptions: bool
    ):
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

                    await bounded_gather2(
                        sema,
                        *[
                            functools.partial(self.copy_source, sema, transfer, r, s, dest_type_task, return_exceptions)
                            for r, s in zip(src_report, src)
                        ],
                        cancel_on_error=True,
                    )

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

    async def _copy(
        self,
        sema: asyncio.Semaphore,
        copy_report: CopyReport,
        transfer: Union[Transfer, List[Transfer]],
        return_exceptions: bool,
    ):
        transfer_report = copy_report._transfer_report
        try:
            if isinstance(transfer, Transfer):
                assert isinstance(transfer_report, TransferReport)
                await self._copy_one_transfer(sema, transfer_report, transfer, return_exceptions)
                return

            assert isinstance(transfer_report, list)
            await bounded_gather2(
                sema,
                *[
                    functools.partial(self._copy_one_transfer, sema, r, t, return_exceptions)
                    for r, t in zip(transfer_report, transfer)
                ],
                return_exceptions=return_exceptions,
                cancel_on_error=True,
            )
        except Exception as e:
            if return_exceptions:
                copy_report.set_exception(e)
            else:
                raise e
