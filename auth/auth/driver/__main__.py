# ruff: noqa: E402
from hailtop.hail_logging import configure_logging

# configure logging before importing anything else
configure_logging()

import asyncio

from .driver import async_main

loop = asyncio.get_event_loop()
loop.run_until_complete(async_main())
