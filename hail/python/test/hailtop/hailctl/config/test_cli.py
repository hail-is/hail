import os

import pytest
from typer.testing import CliRunner

from hailtop.config.user_config import get_user_config_path
from hailtop.config.variables import ConfigVariable
from hailtop.hailctl.config import cli, config_variables


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
        ('domain', 'azure.hail.is'),
        ('gcs_requester_pays/project', 'hail-vdc'),
        ('gcs_requester_pays/buckets', 'hail,foo'),
        ('batch/backend', 'service'),
        ('batch/billing_project', 'test'),
        ('batch/regions', 'us-central1,us-east1'),
        ('batch/remote_tmpdir', 'gs://foo/bar'),
        ('query/backend', 'spark'),
        ('query/jar_url', 'gs://foo/bar.jar'),
        ('query/batch_driver_cores', '1'),
        ('query/batch_worker_cores', '1'),
        ('query/batch_driver_memory', '1Gi'),
        ('query/batch_worker_memory', 'standard'),
        ('query/name_prefix', 'foo'),
        ('query/disable_progress_bar', '1'),
    ],
)
def test_config_set_get_list_unset(name: str, value: str, runner: CliRunner):
    runner.invoke(cli.app, ['set', name, value], catch_exceptions=False)

    res = runner.invoke(cli.app, 'list', catch_exceptions=False)
    assert res.exit_code == 0
    if '/' not in name:
        name = f'global/{name}'
    assert res.stdout.strip() == f'{name}={value}'

    res = runner.invoke(cli.app, ['get', name], catch_exceptions=False)
    assert res.exit_code == 0
    assert res.stdout.strip() == value

    res = runner.invoke(cli.app, ['unset', name], catch_exceptions=False)
    assert res.exit_code == 0

    res = runner.invoke(cli.app, 'list', catch_exceptions=False)
    assert res.exit_code == 0
    assert res.stdout.strip() == ''


# backwards compatibility
def test_config_get_unknown_names(runner: CliRunner, config_dir: str):
    config_path = get_user_config_path(_config_dir=config_dir)
    os.makedirs(os.path.dirname(config_path))
    with open(config_path, 'w', encoding='utf-8') as config:
        config.write(
            '''
[global]
email = johndoe@gmail.com

[batch]
foo = 5
'''
        )

    res = runner.invoke(cli.app, ['get', 'email'], catch_exceptions=False)
    assert res.exit_code == 0
    assert res.stdout.strip() == 'johndoe@gmail.com'

    res = runner.invoke(cli.app, ['get', 'batch/foo'], catch_exceptions=False)
    assert res.exit_code == 0
    assert res.stdout.strip() == '5'


@pytest.mark.parametrize(
    'name,value',
    [
        ('foo/bar', 'baz'),
    ],
)
def test_config_set_unknown_name(name: str, value: str, runner: CliRunner):
    res = runner.invoke(cli.app, ['set', name, value], catch_exceptions=False)
    assert res.exit_code == 2


def test_config_unset_unknown_name(runner: CliRunner):
    # backwards compatibility
    res = runner.invoke(cli.app, ['unset', 'foo'], catch_exceptions=False)
    assert res.exit_code == 0

    res = runner.invoke(cli.app, ['unset', 'foo/bar'], catch_exceptions=False)
    assert res.exit_code == 0


@pytest.mark.parametrize(
    'name,value',
    [
        ('domain', 'foo'),
        ('gcs_requester_pays/project', 'gs://foo/bar'),
        ('gcs_requester_pays/buckets', 'gs://foo/bar'),
        ('batch/backend', 'foo'),
        ('batch/billing_project', 'gs://foo/bar'),
        ('batch/remote_tmpdir', 'asdf://foo/bar'),
        ('query/backend', 'random_backend'),
        ('query/jar_url', 'bar://foo/bar.jar'),
        ('query/batch_driver_cores', 'a'),
        ('query/batch_worker_cores', 'b'),
        ('query/batch_driver_memory', '1bar'),
        ('query/batch_worker_memory', 'random'),
        ('query/disable_progress_bar', '2'),
    ],
)
def test_config_set_bad_value(name: str, value: str, runner: CliRunner):
    res = runner.invoke(cli.app, ['set', name, value], catch_exceptions=False)
    assert res.exit_code == 1


def test_all_config_variables_in_map():
    for variable in ConfigVariable:
        assert variable in config_variables.config_variables()
