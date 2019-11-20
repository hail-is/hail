import asyncio

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
