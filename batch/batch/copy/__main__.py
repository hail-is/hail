from typing import Union, List, Optional, Type
from types import TracebackType
import sys
import json
import asyncio
from concurrent.futures import ThreadPoolExecutor
from hailtop.aiotools.fs import RouterAsyncFS, LocalAsyncFS, Transfer
from hailtop.aiogoogle import GoogleStorageAsyncFS


# This is necessary for the bootsrap process to work.  The bootstrap
# executes build.yaml deploy in local mode to bootstrap the services,
# and no Google credentials are available in the copy steps.
class GoogleAsyncFSIfNecessary:
    def __init__(self, requester_pays_project: Optional[str], transfer: Union[Transfer, List[Transfer]]):
        self.requester_pays_project = requester_pays_project
        self.transfer = transfer
        self.gs = None

    def _requires_gs(self):
        transfers = self.transfer
        if isinstance(transfers, Transfer):
            transfers = [transfers]
        for transfer in transfers:
            if transfer.dest.startswith('gs://'):
                return True
            srcs = transfer.src
            if isinstance(srcs, str):
                srcs = [srcs]
            for src in srcs:
                if src.startswith('gs://'):
                    return True

        return False

    async def __aenter__(self) -> Optional[GoogleStorageAsyncFS]:
        if not self._requires_gs():
            return None

        requester_pays_project = self.requester_pays_project
        if requester_pays_project:
            params = {'userProject': requester_pays_project}
        else:
            params = None
        gs = GoogleStorageAsyncFS(params=params)
        self.gs = gs
        return gs

    async def __aexit__(self,
                        exc_type: Optional[Type[BaseException]],
                        exc_val: Optional[BaseException],
                        exc_tb: Optional[TracebackType]) -> None:
        if self.gs:
            await self.gs.close()


async def copy(requester_pays_project: Optional[str], transfer: Union[Transfer, List[Transfer]]) -> None:
    with ThreadPoolExecutor() as thread_pool:
        async with GoogleAsyncFSIfNecessary(requester_pays_project, transfer) as gs:
            filesystems = [LocalAsyncFS(thread_pool)]
            if gs:
                filesystems.append(gs)
            async with RouterAsyncFS('file', filesystems) as fs:
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
