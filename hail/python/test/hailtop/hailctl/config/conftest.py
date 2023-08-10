import os
import pytest
import tempfile

from typer.testing import CliRunner

from hailtop.hailctl.config.cli import _set


@pytest.fixture()
def config_dir():
    with tempfile.TemporaryDirectory() as d:
        yield d


@pytest.fixture
def runner(config_dir):
    yield CliRunner(mix_stderr=False, env={'XDG_CONFIG_HOME': config_dir})


@pytest.fixture
def bc_runner(config_dir):
    # necessary for backwards compatibility test
    os.environ['XDG_CONFIG_HOME'] = config_dir
    _set('global', 'email', 'johndoe@gmail.com')
    _set('batch', 'foo', '5')

    yield CliRunner(mix_stderr=False, env={'XDG_CONFIG_HOME': config_dir})
