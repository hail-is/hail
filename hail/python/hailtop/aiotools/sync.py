from typing import Optional
import asyncio
import os
import functools

from .copy import GrowingSempahore
from .fs.fs import MultiPartCreate
from .fs.copier import Copier
from .fs.exceptions import UnexpectedEOFError
from .router_fs import RouterAsyncFS, AsyncFS
from ..utils import retry_transient_errors, bounded_gather2
from ..utils.rich_progress_bar import make_listener, CopyToolProgressBar


class SyncError(ValueError):
    pass


async def _copy_file_one_part(
    fs: RouterAsyncFS,
    srcfile: str,
    size: int,
    destfile: str,
    files_listener,
    bytes_listener,
) -> None:
    assert not destfile.endswith('/')

    total_written = 0
    async with await fs.open(srcfile) as srcf:
        try:
            dest_cm = await fs.create(destfile, retry_writes=False)
        except FileNotFoundError:
            await fs.makedirs(os.path.dirname(destfile), exist_ok=True)
            dest_cm = await fs.create(destfile)

        async with dest_cm as destf:
            while True:
                b = await srcf.read(Copier.BUFFER_SIZE)
                if not b:
                    files_listener(-1)
                    bytes_listener(-total_written)
                    return
                written = await destf.write(b)
                assert written == len(b)
                total_written += written


async def _copy_part(
    fs: RouterAsyncFS,
    part_size: int,
    srcfile: str,
    part_number: int,
    this_part_size: int,
    part_creator: MultiPartCreate,
    bytes_listener,
) -> None:
    total_written = 0
    try:
        async with await fs.open_from(srcfile, part_number * part_size, length=this_part_size) as srcf:
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
                print(f'{srcfile}, part {part_number}, complete')
            print(f'{srcfile}, part {part_number}, complete 2')
        print(f'{srcfile}, part {part_number}, complete 3')
        bytes_listener(-total_written)
    except Exception as exc:
        import traceback

        traceback.format_exc()
        print('error in copy part', exc)


async def _copy_file(
    fs: RouterAsyncFS,
    transfer_sema: asyncio.Semaphore,
    srcfile: str,
    destfile: str,
    files_listener,
    bytes_listener,
):
    srcstat = await fs.statfile(srcfile)

    size = await srcstat.size()
    part_size = fs.copy_part_size(destfile)

    if size <= part_size:
        return await retry_transient_errors(
            _copy_file_one_part, fs, srcfile, size, destfile, files_listener, bytes_listener
        )

    n_parts, rem = divmod(size, part_size)
    if rem:
        n_parts += 1

    try:
        part_creator = await fs.multi_part_create(transfer_sema, destfile, n_parts)
    except FileNotFoundError:
        await fs.makedirs(os.path.dirname(destfile), exist_ok=True)
        part_creator = await fs.multi_part_create(transfer_sema, destfile, n_parts)

    async with part_creator:

        async def f(i):
            this_part_size = rem if i == n_parts - 1 and rem else part_size
            await retry_transient_errors(
                _copy_part, fs, part_size, srcfile, i, this_part_size, part_creator, bytes_listener
            )

        await bounded_gather2(transfer_sema, *[functools.partial(f, i) for i in range(n_parts)])
        files_listener(-1)


async def sync(
    plan_folder: str,
    gcs_requester_pays_project: Optional[str],
    verbose: bool,
    max_parallelism: int,
) -> None:
    gcs_kwargs = {'gcs_requester_pays_configuration': gcs_requester_pays_project}
    s3_kwargs = {'max_pool_connections': max_parallelism * 5, 'max_workers': max_parallelism}

    async with RouterAsyncFS(gcs_kwargs=gcs_kwargs, s3_kwargs=s3_kwargs) as fs:
        if not all(
            await asyncio.gather(
                *(
                    fs.exists(os.path.join(plan_folder, x))
                    for x in ('matches', 'differs', 'srconly', 'dstonly', 'plan', 'summary')
                )
            )
        ):
            raise SyncError('Run hailctl fs sync --make-plan first.', 1)
        results = (await fs.read(os.path.join(plan_folder, 'summary'))).decode('utf-8')
        n_files, n_bytes = (int(x) for x in results.split('\t'))
        with CopyToolProgressBar(transient=True, disable=not verbose) as progress:
            files_tid = progress.add_task(description='files', total=n_files, visible=verbose)
            bytes_tid = progress.add_task(description='bytes', total=n_bytes, visible=verbose)

            files_listener = make_listener(progress, files_tid)
            bytes_listener = make_listener(progress, bytes_tid)

            max_file_parallelism = max(1, max_parallelism // 10)

            initial_parallelism = min(10, max_parallelism)
            initial_file_parallelism = min(10, max_file_parallelism)

            parallelism_tid = progress.add_task(
                description='transfer parallelism',
                completed=initial_parallelism,
                total=max_parallelism,
                visible=verbose,
            )
            file_parallelism_tid = progress.add_task(
                description='file parallelism',
                completed=initial_parallelism,
                total=max_parallelism,
                visible=verbose,
            )

            async with GrowingSempahore(1, 1, (progress, parallelism_tid)) as file_sema, GrowingSempahore(
                initial_file_parallelism, max_file_parallelism, (progress, file_parallelism_tid)
            ) as transfer_sema:
                await bounded_gather2(
                    file_sema,
                    *[
                        functools.partial(_copy_file, fs, transfer_sema, src, dst, files_listener, bytes_listener)
                        async for src, dst in iterate_plan_file(plan_folder, fs)
                    ],
                )


async def iterate_plan_file(plan_folder: str, fs: AsyncFS):
    lineno = 0
    plan = (await fs.read(os.path.join(plan_folder, 'plan'))).decode('utf-8')
    for line in plan.split('\n'):
        if not line:
            continue
        parts = line.strip().split('\t')
        if len(parts) != 2:
            raise SyncError(f'Malformed plan line, {lineno}, must have exactly one tab: {line}', 1)
        yield parts
