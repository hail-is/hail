import orjson
from typer.testing import CliRunner

from hailtop.hailctl.dev import config as cli

default_config = {
    'domain': 'example.com',
    'location': 'external',
    'default_namespace': 'default',
}


def set_deploy_config(deploy_config_file: str, config: dict):
    with open(deploy_config_file, 'w', encoding='utf-8') as f:
        f.write(orjson.dumps(config).decode('utf-8'))


def load_deploy_config_dict(deploy_config_file: str) -> dict:
    with open(deploy_config_file, 'r', encoding='utf-8') as f:
        return orjson.loads(f.read())


def test_dev_config_set(runner: CliRunner, deploy_config_file: str):
    set_deploy_config(deploy_config_file, default_config)

    res = runner.invoke(cli.app, ['set', 'default_namespace', 'foo'])
    assert res.exit_code == 0, res.stderr

    expected = {'domain': 'example.com', 'location': 'external', 'default_namespace': 'foo'}
    assert load_deploy_config_dict(deploy_config_file) == expected


def test_dev_config_set_not_affected_by_env_vars(runner: CliRunner, deploy_config_file: str):
    set_deploy_config(deploy_config_file, default_config)

    res = runner.invoke(
        cli.app,
        ['set', 'default_namespace', 'foo'],
        env={'HAIL_DOMAIN': 'foo.example.com'},
    )
    assert res.exit_code == 0

    expected = {'domain': 'example.com', 'location': 'external', 'default_namespace': 'foo'}
    assert load_deploy_config_dict(deploy_config_file) == expected
