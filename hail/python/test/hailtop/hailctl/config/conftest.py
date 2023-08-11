import os
import pytest
import tempfile

from typer.testing import CliRunner


@pytest.fixture()
def config_dir():
    with tempfile.TemporaryDirectory() as d:
        yield d


@pytest.fixture
def runner(config_dir):
    yield CliRunner(mix_stderr=False, env={'XDG_CONFIG_HOME': config_dir})


@pytest.fixture
def bc_runner(config_dir):
    from hailtop.config import get_user_config, get_user_config_path  # pylint: disable=import-outside-toplevel

    # necessary for backwards compatibility test
    os.environ['XDG_CONFIG_HOME'] = config_dir

    config = get_user_config()
    config_file = get_user_config_path()

    items = [
        ('global', 'email', 'johndoe@gmail.com'),
        ('batch', 'foo', '5')
    ]

    for section, key, value in items:
        if section not in config:
            config[section] = {}
        config[section][key] = value

        try:
            f = open(config_file, 'w', encoding='utf-8')
        except FileNotFoundError:
            os.makedirs(config_file.parent, exist_ok=True)
            f = open(config_file, 'w', encoding='utf-8')
        with f:
            config.write(f)

    yield CliRunner(mix_stderr=False, env={'XDG_CONFIG_HOME': config_dir})
