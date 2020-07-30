from hailtop.hail_logging import configure_logging
# configure logging before importing anything else
configure_logging()

import asyncio  # noqa: E402 pylint: disable=wrong-import-position
from .driver import async_main  # noqa: E402 pylint: disable=wrong-import-position

loop = asyncio.get_event_loop()
loop.run_until_complete(async_main())
