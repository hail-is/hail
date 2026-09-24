# ruff: noqa: E402
from hailtop.hail_logging import configure_logging

# configure logging before importing anything else
configure_logging()

import asyncio

from .driver import async_main

loop = asyncio.new_event_loop()
asyncio.set_event_loop(loop)
loop.run_until_complete(async_main())
