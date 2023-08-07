import pytest

from typer.testing import CliRunner

from hailtop.hailctl.config import cli


def test_config_location(runner: CliRunner, config_dir: str):
    res = runner.invoke(cli.app, 'config-location', catch_exceptions=False)
    assert res.exit_code == 0
    assert res.stdout.strip() == f'{config_dir}/hail/config.ini'


def test_config_list_empty_config(runner: CliRunner):
    res = runner.invoke(cli.app, 'list', catch_exceptions=False)
    assert res.exit_code == 0
    assert res.stdout.strip() == ''


@pytest.mark.parametrize(
    'name,value',
    [
        ('batch/backend', 'batch'),
        ('batch/billing_project', 'test'),
        ('batch/remote_tmpdir', 'gs://foo/bar'),
        ('query/backend', 'spark'),

        # hailctl currently accepts arbitrary settings
        ('foo/bar', 'baz'),
    ],
)
def test_config_set(name: str, value: str, runner: CliRunner):
    runner.invoke(cli.app, ['set', name, value], catch_exceptions=False)

    res = runner.invoke(cli.app, 'list', catch_exceptions=False)
    assert res.exit_code == 0
    assert res.stdout.strip() == f'{name}={value}'

    res = runner.invoke(cli.app, ['get', name], catch_exceptions=False)
    assert res.exit_code == 0
    assert res.stdout.strip() == value


def test_config_get_bad_names(runner: CliRunner):
    res = runner.invoke(cli.app, ['get', 'foo'], catch_exceptions=False)
    assert res.exit_code == 0
    assert res.stdout.strip() == ''

    res = runner.invoke(cli.app, ['get', '/a/b/c'], catch_exceptions=False)
    assert res.exit_code == 1


@pytest.mark.parametrize(
    'name,value',
    [
        ('batch/remote_tmpdir', 'asdf://foo/bar'),
    ],
)
def test_config_set_bad_value(name: str, value: str, runner: CliRunner):
    res = runner.invoke(cli.app, ['set', name, value], catch_exceptions=False)
    assert res.exit_code == 1
