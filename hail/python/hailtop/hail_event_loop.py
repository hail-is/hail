import asyncio

def hail_event_loop():
    '''If a running event loop exists, use nest_asyncio to allow Hail's event loops to nest inside
    it.
    If no event loop exists, ask asyncio to get one for us.
    '''
    import asyncio  # pylint: disable=import-outside-toplevel
    import nest_asyncio  # pylint: disable=import-outside-toplevel

    try:
        asyncio.get_running_loop()
        nest_asyncio.apply()
        return asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.get_event_loop()
