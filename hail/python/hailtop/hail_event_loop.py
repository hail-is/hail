import asyncio

import nest_asyncio


def hail_event_loop() -> asyncio.AbstractEventLoop:
    '''If a running event loop exists, use nest_asyncio to allow Hail's event loops to nest inside
    it.
    If no event loop exists, ask asyncio to get one for us.
    '''

    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        return loop
    nest_asyncio.apply(loop)
    return loop
