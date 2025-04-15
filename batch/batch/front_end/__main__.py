# ruff: noqa: E402
from hailtop.hail_logging import configure_logging

# configure logging before importing anything else
configure_logging()

from .front_end import run

run()
