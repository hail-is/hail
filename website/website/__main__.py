import sys
from hailtop.hail_logging import configure_logging

# configure logging before importing anything else
configure_logging()

from .website import run  # noqa: E402 pylint: disable=wrong-import-position

local_mode = False

if len(sys.argv) > 1:
    assert len(sys.argv) == 2 and sys.argv[1] == 'local', sys.argv
    local_mode = True

run(local_mode=local_mode)
