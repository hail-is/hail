# ruff: noqa: E402
from hailtop.hail_logging import configure_logging

# configure logging before importing anything else
configure_logging()

import sys
import traceback

import aiohttp

from .main import run

oldinit = aiohttp.ClientSession.__init__  # type: ignore


def newinit(self, *args, **kwargs):
    oldinit(self, *args, **kwargs)
    self._source_traceback = traceback.extract_stack(sys._getframe(1))


aiohttp.ClientSession.__init__ = newinit  # type: ignore


run()
