import asyncio
import aiohttp

from .logging import log


async def retry_forever(f, msg=None):
    i = 0
    while True:
        try:
            await f()
            break
        except Exception as exc:
            if msg:
                log.warning(msg(exc))
        i = 1.1 ** i
        await asyncio.sleep(0.100 * i)
        # max 44.5s
        if i < 64:
            i = i + 1


async def retry_aiohttp_forever(f):
    i = 0
    while True:
        try:
            await f()
            break
        except (aiohttp.client_exceptions.ClientResponseError,
                aiohttp.client_exceptions.ClientOSError,
                aiohttp.client_exceptions.ClientConnectorError) as exc:
            log.warning(f'backing off due to {exc}')
        i = 1.1 ** i
        await asyncio.sleep(0.100 * i)
        # max 44.5s
        if i < 64:
            i = i + 1
