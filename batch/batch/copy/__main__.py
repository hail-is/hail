from typing import Union, List, Optional
import sys
import json
import asyncio
from concurrent.futures import ThreadPoolExecutor
from hailtop.aiotools.fs import RouterAsyncFS, LocalAsyncFS, Transfer
from hailtop.aiogoogle import GoogleStorageAsyncFS


async def copy(requester_pays_project: Optional[str], transfer: Union[Transfer, List[Transfer]]) -> None:
    if requester_pays_project:
        params = {'userProject': requester_pays_project}
    else:
        params = None
    with ThreadPoolExecutor() as thread_pool:
        async with RouterAsyncFS('file', [LocalAsyncFS(thread_pool), GoogleStorageAsyncFS(params=params)]) as fs:
            sema = asyncio.Semaphore(50)
            async with sema:
                copy_report = await fs.copy(sema, transfer)
                copy_report.summarize()


async def main() -> None:
    assert len(sys.argv) == 3
    requster_pays_project = json.loads(sys.argv[1])
    files = json.loads(sys.argv[2])

    await copy(
        requster_pays_project, [Transfer(f['from'], f['to'], treat_dest_as=Transfer.DEST_IS_TARGET) for f in files]
    )


if __name__ == '__main__':
    asyncio.run(main())
