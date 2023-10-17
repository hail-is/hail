from typing import Optional
import asyncio
import os
import sys

from .fs.copier import Transfer
from .router_fs import RouterAsyncFS, AsyncFS
from .copy import copy

try:
    import uvloop

    uvloop_install = uvloop.install
except ImportError as e:
    if not sys.platform.startswith('win32'):
        raise e

    def uvloop_install():
        pass


class SyncError(ValueError):
    pass


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
        await copy(
            max_simultaneous_transfers=max_parallelism,
            local_kwargs=None,
            gcs_kwargs=gcs_kwargs,
            azure_kwargs={},
            s3_kwargs=s3_kwargs,
            transfers=[
                Transfer(src, dst, treat_dest_as=Transfer.DEST_IS_TARGET)
                async for src, dst in iterate_plan_file(plan_folder, fs)
            ],
            verbose=verbose,
            totals=(n_files, n_bytes),
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
