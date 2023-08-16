from hailtop.hail_logging import configure_logging

# configure logging before importing anything else
configure_logging()

import sys  # noqa: E402 pylint: disable=wrong-import-position
import traceback  # noqa: E402 pylint: disable=wrong-import-position

import aiohttp  # noqa: E402 pylint: disable=wrong-import-position

from .main import run  # noqa: E402 pylint: disable=wrong-import-position

oldinit = aiohttp.ClientSession.__init__


def newinit(self, *args, **kwargs):
    oldinit(self, *args, **kwargs)
    self._source_traceback = traceback.extract_stack(sys._getframe(1))


aiohttp.ClientSession.__init__ = newinit


run()
