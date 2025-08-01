import tempfile

import pytest
from typer.testing import CliRunner


@pytest.fixture()
def config_dir():
    with tempfile.TemporaryDirectory() as d:
        yield d


@pytest.fixture
def runner(config_dir):
    yield CliRunner(env={'XDG_CONFIG_HOME': config_dir})
