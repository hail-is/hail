import logging
import pytest

log = logging.getLogger(__name__)

@pytest.fixture(autouse=True)
def log_before_after():
    log.info('starting test')
    yield
    log.info('ending test')
