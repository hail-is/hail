from gear import configure_logging
# configure logging before importing anything else
configure_logging()

from .scorecard import run  # noqa: E402

run()
