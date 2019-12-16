from gear import configure_logging
# configure logging before importing anything else
configure_logging()


def main():
    import asyncio
    from .driver import async_main

    loop = asyncio.get_event_loop()
    loop.run_until_complete(async_main())


main()
