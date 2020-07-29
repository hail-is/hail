from hailtop.hail_logging import configure_logging
configure_logging()

from .notebook import run  # noqa: E402 pylint: disable=wrong-import-position

run()
