from typing import Union, List, Optional
import asyncio
from concurrent.futures import ThreadPoolExecutor
from hailtop.aiotools.fs import AsyncFS, RouterAsyncFS, LocalAsyncFS, Transfer
from hailtop.aiogoogle import GoogleStorageAsyncFS


async def copy(requster_pays_project: Optional[str], transfer: Union[Transfer, List[Transfer]]) -> None:
    with ThreadPoolExecutor() as thread_pool:
        fs = RouterAsyncFS(
            'file', [LocalAsyncFS(thread_pool), GoogleStorageAsyncFS(requester_pays_project=requester_pays_project)])
        sema = asyncio.Semaphore(50)
        async with sema:
            await fs.copy(sema, transfers)


def main() -> None:
    assert len(sys.argv) == 3
    requster_pays_project = json.loads(sys.argv[1])
    files = json.loads(sys.argv[2])

    await copy(requster_pays_project, [
        Trasnfer(f['from'], f['to'], treat_test_as=Transfer.DEST_IS_TARGET)
        for f in files])


if __name__ == '__main__':
    asyncio.run(main())
