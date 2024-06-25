import tempfile

import pytest
from typer.testing import CliRunner


@pytest.fixture()
def deploy_config_file():
    with tempfile.NamedTemporaryFile() as f:
        yield f.name


@pytest.fixture
def runner(deploy_config_file):
    yield CliRunner(mix_stderr=False, env={'HAIL_DEPLOY_CONFIG_FILE': deploy_config_file})
