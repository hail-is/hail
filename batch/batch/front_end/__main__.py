from hailtop.hail_logging import configure_logging
# configure logging before importing anything else
configure_logging()

from .front_end import run  # noqa: E402 pylint: disable=wrong-import-position

run()
