import sys
import time
import functools
import logging
import asyncio
import aiohttp
import hailtop.aiogoogle as aiogoogle
import hailtop.aiogoogle.container

log = logging.getLogger(__name__)


class AsyncIOExecutor:
    def __init__(self, parallelism, queue_size=10):
        self._queue = asyncio.Queue(maxsize=queue_size)
        for _ in range(parallelism):
            asyncio.ensure_future(self._worker())

    async def _worker(self):
        while True:
            fut, f, args, kwargs = await self._queue.get()
            try:
                fut.set_result(await f(*args, **kwargs))
            except asyncio.CancelledError:  # pylint: disable=try-except-raise
                raise
            except Exception as e:  # pylint: disable=broad-except
                fut.set_exception(e)

    async def submit(self, f, *args, **kwargs):
        fut = asyncio.Future()
        await self._queue.put((fut, f, args, kwargs))
        return fut

    async def gather(self, pfs):
        futs = [await self.submit(pf) for pf in pfs]
        return [await fut for fut in futs]


class CleanupImages:
    def __init__(self, client):
        self._executor = AsyncIOExecutor(8)
        self._client = client

    async def cleanup_digest(self, image, digest, tags):
        log.info(f'cleaning up digest {image}@{digest}')

        await self._executor.gather([
            functools.partial(self._client.delete_image_tag, image, tag)
            for tag in tags])

        await (await self._executor.submit(self._client.delete_image, image, digest))

        log.info(f'cleaned up digest  {image}@{digest}')

    async def cleanup_image(self, image):
        log.info(f'cleaning up image {image}')

        log.info(f'listing tags for {image}')

        result = await (await self._executor.submit(self._client.list_image_tags, image))
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
            if (now - time_uploaded) >= (7 * 24 * 60 * 60) or len(tags) == 0])

        log.info(f'cleaned up image  {image}')

    async def run(self):
        images = await (await self._executor.submit(self._client.list_images))
        await asyncio.gather(*[
            self.cleanup_image(image)
            for image in images['child']
        ])


async def main():
    logging.basicConfig(level=logging.INFO)

    if len(sys.argv) != 2:
        raise ValueError('usage: cleanup_gcr <project>')
    project = sys.argv[1]

    async with aiogoogle.container.Client(
            project=project,
            raise_for_status=True,
            timeout=aiohttp.ClientTimeout(total=60)) as client:
        cleanup_images = CleanupImages(client)
        await cleanup_images.run()


asyncio.run(main())
