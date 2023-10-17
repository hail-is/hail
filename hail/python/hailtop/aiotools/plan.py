from typing import List, Tuple, Optional
import asyncio
import os
import sys
from contextlib import AsyncExitStack
from hailtop.aiotools import FileAndDirectoryError

from .router_fs import RouterAsyncFS
from .fs import FileListEntry, FileStatus, AsyncFS, WritableStream
from ..utils.rich_progress_bar import CopyToolProgressBar, Progress

try:
    import uvloop

    uvloop_install = uvloop.install
except ImportError as e:
    if not sys.platform.startswith('win32'):
        raise e

    def uvloop_install():
        pass


class PlanError(ValueError):
    pass


async def plan(
    folder: str,
    copy_to: List[Tuple[str, str]],
    copy_into: List[Tuple[str, str]],
    gcs_requester_pays_project: Optional[str],
    verbose: bool,
    max_parallelism: int,
):
    if gcs_requester_pays_project:
        gcs_kwargs = {'gcs_requester_pays_configuration': gcs_requester_pays_project}
    else:
        gcs_kwargs = {}

    total_n_files = 0
    total_n_bytes = 0

    async with AsyncExitStack() as cleanups:
        fs = await cleanups.enter_async_context(RouterAsyncFS(gcs_kwargs=gcs_kwargs))
        source_destination_pairs = [
            *copy_to,
            *((src, copied_into_path(fs, source_path=src, folder_path=dst)) for src, dst in copy_into),
        ]

        if any(await asyncio.gather(fs.isfile(folder), fs.isdir(folder.rstrip('/') + '/'))):
            raise PlanError(f'plan folder already exists: {folder}', 1)

        await fs.mkdir(folder)
        matches = await cleanups.enter_async_context(await fs.create(os.path.join(folder, 'matches')))
        differs = await cleanups.enter_async_context(await fs.create(os.path.join(folder, 'differs')))
        srconly = await cleanups.enter_async_context(await fs.create(os.path.join(folder, 'srconly')))
        dstonly = await cleanups.enter_async_context(await fs.create(os.path.join(folder, 'dstonly')))
        plan = await cleanups.enter_async_context(await fs.create(os.path.join(folder, 'plan')))

        progress = cleanups.enter_context(CopyToolProgressBar(transient=True, disable=not verbose))

        for src, dst in source_destination_pairs:
            n_files, n_bytes = await find_all_copy_pairs(
                fs,
                matches,
                differs,
                srconly,
                dstonly,
                plan,
                src,
                dst,
                progress,
                asyncio.Semaphore(max_parallelism),
                source_must_exist=True,
            )
            total_n_files += n_files
            total_n_bytes += n_bytes

        summary = await cleanups.enter_async_context(await fs.create(os.path.join(folder, 'summary')))
        await summary.write((f'{total_n_files}\t{total_n_bytes}\n').encode('utf-8'))


def copied_into_path(fs: AsyncFS, *, source_path: str, folder_path: str) -> str:
    src_url = fs.parse_url(source_path)
    dest_url = fs.parse_url(folder_path)
    src_basename = os.path.basename(src_url.path.rstrip('/'))
    return str(dest_url.with_new_path_component(src_basename))


class FileStat:
    def __init__(self, basename, url, is_dir, size):
        self.basename = basename
        self.url = url
        self.is_dir = is_dir
        self.size = size

    def __repr__(self) -> str:
        return f'FileStat({self.basename}, {self.url}, {self.is_dir}, {self.size})'

    @staticmethod
    async def from_file_list_entry(x: FileListEntry) -> 'FileStat':
        url, is_dir = await asyncio.gather(x.url(), x.is_dir())
        if is_dir:
            size = 0
        else:
            size = await (await x.status()).size()
        return FileStat(x.basename(), url, is_dir, size)

    @staticmethod
    async def from_file_status(x: FileStatus) -> 'FileStat':
        return FileStat(x.basename(), x.url(), False, await x.size())


async def listfiles(fs: AsyncFS, x: str) -> List[FileStat]:
    try:
        it = await fs.listfiles(x)
        return [await FileStat.from_file_list_entry(x) async for x in it]
    except (FileNotFoundError, NotADirectoryError):
        return []


async def statfile(fs: AsyncFS, x: str) -> Optional[FileStat]:
    try:
        file_status = await fs.statfile(x)
        return await FileStat.from_file_status(file_status)
    except FileNotFoundError:
        return None


async def writeline(file: WritableStream, *columns: str):
    await file.write(('\t'.join(columns) + '\n').encode('utf-8'))


def relativize_url(folder: str, file: str) -> str:
    if folder[-1] != '/':
        folder = folder + '/'
    relative_path = file.removeprefix(folder)
    assert relative_path[0] != '/'
    return relative_path


async def find_all_copy_pairs(
    fs: AsyncFS,
    matches: WritableStream,
    differs: WritableStream,
    srconly: WritableStream,
    dstonly: WritableStream,
    plan: WritableStream,
    src: str,
    dst: str,
    progress: Progress,
    sema: asyncio.Semaphore,
    source_must_exist: bool,
) -> Tuple[int, int]:
    async with sema:
        srcstat, srcfiles, dststat, dstfiles = await asyncio.gather(
            statfile(fs, src),
            listfiles(fs, src),
            statfile(fs, dst),
            listfiles(fs, dst),
        )

        if srcstat:
            if srcstat.url[-1] == '/':
                if srcstat.size == 0:
                    srcstat = None
                    if srcfiles:
                        print(f'Not copying size-zero source file which shares a name with a source directory. {src}')
                    else:
                        print(f'Not copying size-zero source file whose name looks like a directory. {src}')
                else:
                    raise PlanError(
                        f'Source is a non-size-zero file whose name looks like a directory. This is not supported. {src} {srcstat}',
                        1,
                    )
            elif srcfiles:
                raise PlanError(
                    f'Source is both a directory and a file (other than a size-zero file whose name ends in "/"). '
                    f'This is not supported. {src} {srcstat}',
                    1,
                ) from FileAndDirectoryError(src)
        if dststat:
            if dststat.url[-1] == '/':
                if dststat.size == 0:
                    dststat = None
                    if dstfiles:
                        print(
                            f'Ignoring size-zero destination file which shares a name with a destination directory. {dst}'
                        )
                    else:
                        print(f'Ignoring size-zero destination file whose name looks like a directory. {dst}')
                else:
                    raise PlanError(
                        f'Destination is a non-size-zero file whose name looks like a directory. This is not supported. {dst} {dststat}',
                        1,
                    )
            elif dstfiles:
                raise PlanError(
                    f'Destination identifies both a directory and a file (other than a size-zero file whose name ends '
                    f'in "/"). This is not supported. {dst} {dststat}',
                    1,
                ) from FileAndDirectoryError(dst)
        if srcstat and dstfiles:
            raise PlanError(
                f'Source is a file but destination is a directory. This is not supported. {src} -> {dst}', 1
            ) from IsADirectoryError(dst)
        if srcfiles and dststat:
            raise PlanError(
                f'Source is a directory but destination is a file. This is not supported. {src} -> {dst}', 1
            ) from IsADirectoryError(src)
        if source_must_exist and not srcstat and not srcfiles:
            raise PlanError(f'Source is neither a folder nor a file: {src}', 1) from FileNotFoundError(src)

        if srcstat:
            assert len(srcfiles) == 0
            assert len(dstfiles) == 0

            if dststat:
                if srcstat.size == dststat.size:
                    await writeline(matches, srcstat.url, dststat.url)
                    return 0, 0
                await writeline(differs, srcstat.url, dststat.url, str(srcstat.size), str(dststat.size))
                return 0, 0
            await writeline(srconly, srcstat.url)
            await writeline(plan, srcstat.url, dst)
            return 1, srcstat.size

        srcfiles.sort(key=lambda x: x.basename)
        dstfiles.sort(key=lambda x: x.basename)

        tid = progress.add_task(description=src, total=len(srcfiles) + len(dstfiles))

        srcidx = 0
        dstidx = 0

        n_files = 0
        n_bytes = 0

        async def process_child_directory(new_srcurl: str, new_dsturl: str) -> Tuple[int, int]:
            return await find_all_copy_pairs(
                fs,
                matches,
                differs,
                srconly,
                dstonly,
                plan,
                new_srcurl,
                new_dsturl,
                progress,
                sema,
                source_must_exist=False,
            )

        async def retrieve_child_directory_results(child_directory_task):
            nonlocal n_files
            nonlocal n_bytes
            dir_n_files, dir_n_bytes = await child_directory_task
            n_files += dir_n_files
            n_bytes += dir_n_bytes

        async with AsyncExitStack() as child_directory_callbacks:

            def background_process_child_dir(new_srcurl: str, new_dsturl: str):
                t = asyncio.create_task(process_child_directory(new_srcurl, new_dsturl))
                child_directory_callbacks.push_async_callback(retrieve_child_directory_results, t)

            while srcidx < len(srcfiles) and dstidx < len(dstfiles):
                srcf = srcfiles[srcidx]
                dstf = dstfiles[dstidx]
                if srcf.basename == dstf.basename:
                    if srcf.is_dir and dstf.is_dir:
                        background_process_child_dir(srcf.url, dstf.url)
                    elif srcf.is_dir and not dstf.is_dir:
                        await writeline(differs, srcf.url, dstf.url, 'dir', 'file')
                    elif not srcf.is_dir and dstf.is_dir:
                        await writeline(differs, srcf.url, dstf.url, 'file', 'dir')
                    elif srcf.size == dstf.size:
                        await writeline(matches, srcf.url, dstf.url)
                    else:
                        await writeline(differs, srcf.url, dstf.url, str(srcf.size), str(dstf.size))
                    dstidx += 1
                    srcidx += 1
                    progress.update(tid, advance=2)
                elif srcf.basename < dstf.basename:
                    if srcf.is_dir:
                        background_process_child_dir(
                            srcf.url,
                            os.path.join(dst, relativize_url(folder=src, file=srcf.url)),
                        )
                    else:
                        await writeline(srconly, srcf.url)
                        await writeline(plan, srcf.url, os.path.join(dst, srcf.basename))
                        n_files += 1
                        n_bytes += srcf.size
                    srcidx += 1
                    progress.update(tid, advance=1)
                else:
                    assert srcf.basename >= dstf.basename
                    dstidx += 1
                    progress.update(tid, advance=1)
                    await writeline(dstonly, dstf.url)

            while srcidx < len(srcfiles):
                srcf = srcfiles[srcidx]

                if srcf.is_dir:
                    background_process_child_dir(
                        srcf.url,
                        os.path.join(dst, relativize_url(folder=src, file=srcf.url)),
                    )
                else:
                    await writeline(srconly, srcf.url)
                    await writeline(plan, srcf.url, os.path.join(dst, srcf.basename))
                    n_files += 1
                    n_bytes += srcf.size
                srcidx += 1
                progress.update(tid, advance=1)

            while dstidx < len(dstfiles):
                dstf = dstfiles[dstidx]

                await writeline(dstonly, dstf.url)
                dstidx += 1
                progress.update(tid, advance=1)

        # a short sleep ensures the progress bar is visible for a moment to the user
        await asyncio.sleep(0.150)
        progress.remove_task(tid)

    return n_files, n_bytes
