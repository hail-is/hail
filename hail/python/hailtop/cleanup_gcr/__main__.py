import asyncio
import logging
import sys
import time
from typing import Awaitable, List, TypeVar

import aiohttp

from hailtop import aiotools
from hailtop.aiocloud import aiogoogle

log = logging.getLogger(__name__)


T = TypeVar('T')


class AsyncIOExecutor:
    def __init__(self, parallelism):
        self._semaphore = asyncio.Semaphore(parallelism)
        self.task_manager = aiotools.BackgroundTaskManager()

    def shutdown(self):
        self.task_manager.shutdown()

    async def _run(self, fut, aw):
        async with self._semaphore:
            try:
                fut.set_result(await aw)
            except asyncio.CancelledError:  # pylint: disable=try-except-raise
                raise
            except Exception as e:  # pylint: disable=broad-except
                fut.set_exception(e)

    def submit(self, aw):
        fut = asyncio.Future()  # type: ignore
        self.task_manager.ensure_future(self._run(fut, aw))
        return fut

    async def gather(self, aws: List[Awaitable[T]]) -> List[T]:
        futs = [self.submit(aw) for aw in aws]
        return [await fut for fut in futs]


class CleanupImages:
    def __init__(self, client):
        self._executor = AsyncIOExecutor(8)
        self._client = client

    def shutdown(self):
        self._executor.shutdown()

    async def cleanup_digest(self, image, digest, tags):
        log.info(f'cleaning up digest {image}@{digest}')

        async def delete_tag(tag: str):
            try:
                await self._client.delete(f'/{image}/manifests/{tag}')
            except aiohttp.ClientResponseError as e:
                if e.status != 404:
                    raise

        await self._executor.gather([delete_tag(tag) for tag in tags])

        try:
            await self._executor.submit(self._client.delete(f'/{image}/manifests/{digest}'))
        except aiohttp.ClientResponseError as e:
            if e.status != 404:
                raise

        log.info(f'cleaned up digest  {image}@{digest}')

    async def cleanup_image(self, image):
        log.info(f'cleaning up image {image}')

        log.info(f'listing tags for {image}')

        result = await self._executor.submit(self._client.get(f'/{image}/tags/list'))
        manifests = result['manifest']
        manifests = [(digest, int(data['timeUploadedMs']) / 1000, data['tag']) for digest, data in manifests.items()]

        log.info(f'got {len(manifests)} manifests for {image}')

        # sort is ascending, oldest first
        manifests = sorted(manifests, key=lambda x: x[1])

        # keep the most recent 10
        manifests = manifests[:-10]

        now = time.time()
        await asyncio.gather(*[
            self.cleanup_digest(image, digest, tags)
            for digest, time_uploaded, tags in manifests
            if (now - time_uploaded) >= (7 * 24 * 60 * 60) or len(tags) == 0
        ])

        log.info(f'cleaned up image  {image}')

    async def run(self):
        images = await self._executor.submit(self._client.get('/tags/list'))
        await asyncio.gather(*[self.cleanup_image(image) for image in images['child']])


async def main():
    logging.basicConfig(level=logging.INFO)

    if len(sys.argv) != 2:
        raise ValueError('usage: cleanup_gcr <project>')
    project = sys.argv[1]

    async with aiogoogle.GoogleContainerClient(project=project, timeout=aiohttp.ClientTimeout(total=5)) as client:
        cleanup_images = CleanupImages(client)
        try:
            await cleanup_images.run()
        finally:
            cleanup_images.shutdown()


asyncio.run(main())
