import os

import pytest
from typer.testing import CliRunner

from hailtop.config.user_config import get_user_config_path
from hailtop.config.variables import ConfigVariable
from hailtop.hailctl.config import cli, config_variables


def test_config_location(runner: CliRunner, config_dir: str):
    res = runner.invoke(cli.app, 'config-location', catch_exceptions=False)
    assert res.exit_code == 0
    assert res.stdout.strip() == f'Default settings: {config_dir}/hail/config.ini'


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
        config.write("""
[global]
email = johndoe@gmail.com

[batch]
foo = 5
""")

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


def test_profile(runner: CliRunner, config_dir: str):
    os.remove(f'{config_dir}/hail/config.ini')

    # check there are no variables set
    res = runner.invoke(cli.app, 'list', catch_exceptions=False)
    assert res.exit_code == 0
    assert res.stdout.strip() == ''

    orig_remote_tmpdir = 'gs://my-bucket/tmp/batch/'
    new_remote_tmpdir = 'gs://my-bucket-2/tmp/batch/'
    default_backend = 'spark'

    # set default variables
    runner.invoke(cli.app, ['set', 'batch/remote_tmpdir', orig_remote_tmpdir], catch_exceptions=False)
    runner.invoke(cli.app, ['set', 'query/backend', default_backend], catch_exceptions=False)

    # list the default config variables
    res = runner.invoke(cli.app, 'list', catch_exceptions=False)
    assert res.exit_code == 0
    assert res.stdout.strip() == f'Config settings from {config_dir}/hail/config.ini:\n\nbatch/remote_tmpdir={orig_remote_tmpdir}\nquery/backend={default_backend}\n'

    # create a new profile
    res = runner.invoke(cli.profile_app, ['create', 'profile1'], catch_exceptions=False)
    assert res.exit_code == 0

    # make sure the profile is still the default one
    res = runner.invoke(cli.app, ['get', 'global/profile'], catch_exceptions=False)
    assert res.exit_code == 0
    assert res.stdout.strip() == 'default'

    # load the new profile
    res = runner.invoke(cli.profile_app, ['load', 'profile1'], catch_exceptions=False)
    assert res.exit_code == 0

    # Make sure the original config is still there
    res = runner.invoke(cli.app, 'list', catch_exceptions=False)
    assert res.exit_code == 0
    assert res.stdout.strip() == f'Config settings from {config_dir}/hail/config.ini:\n\nbatch/remote_tmpdir={orig_remote_tmpdir}\nquery/backend={default_backend}\nglobal/profile=profile1\n'

    # the value for profile should be the new one (not the default one)
    res = runner.invoke(cli.app, ['get', 'global/profile'], catch_exceptions=False)
    assert res.exit_code == 0
    assert res.stdout.strip() == 'profile1'

    # reset the value of remote_tmpdir to something new
    runner.invoke(cli.app, ['set', 'batch/remote_tmpdir', new_remote_tmpdir], catch_exceptions=False)
    assert res.exit_code == 0

    # List the new config to make sure the defaults are overridden if a profile-specific value exists while keeping additional defaults
    res = runner.invoke(cli.app, 'list', catch_exceptions=False)
    assert res.exit_code == 0
    assert res.stdout.strip() == f'Config settings from {config_dir}/hail/config.ini:\n\nquery/backend={default_backend}\nglobal/profile=profile1\n\nConfig settings from {config_dir}/hail/profile1.ini:\nbatch/remote_tmpdir={new_remote_tmpdir}\n'

    # List the available profiles
    res = runner.invoke(cli.profile_app, ['list'], catch_exceptions=False)
    assert res.exit_code == 0
    assert res.stdout.strip() == '  default\n* profile1'

    # the value for remote tmpdir should be the new one (not the default one)
    res = runner.invoke(cli.app, ['get', 'batch/remote_tmpdir'], catch_exceptions=False)
    assert res.exit_code == 0
    assert res.stdout.strip() == new_remote_tmpdir

    # Test config location
    res = runner.invoke(cli.app, 'config-location', catch_exceptions=False)
    assert res.exit_code == 0
    assert res.stdout.strip() == f'Default settings: {config_dir}/hail/config.ini\nOverrode default settings with profile "profile1": {config_dir}/hail/profile1.ini'

    # can't delete an active profile
    res = runner.invoke(cli.profile_app, ['delete', 'profile1'], catch_exceptions=False)
    assert res.exit_code == 1

    # can't delete the default profile
    res = runner.invoke(cli.profile_app, ['delete', 'default'], catch_exceptions=False)
    assert res.exit_code == 1

    # load the original profile
    res = runner.invoke(cli.profile_app, ['load', 'default'], catch_exceptions=False)
    assert res.exit_code == 0

    # Make sure the original config is still there
    res = runner.invoke(cli.app, 'list', catch_exceptions=False)
    assert res.exit_code == 0
    assert res.stdout.strip() == f'Config settings from {config_dir}/hail/config.ini:\n\nbatch/remote_tmpdir={orig_remote_tmpdir}\nquery/backend={default_backend}\nglobal/profile=profile1\n'

    # delete non-active profile
    res = runner.invoke(cli.profile_app, ['delete', 'profile1'], catch_exceptions=False)
    assert res.exit_code == 0

    # list profiles
    res = runner.invoke(cli.profile_app, ['list'], catch_exceptions=False)
    assert res.exit_code == 0
    assert res.stdout.strip() == '* default'

    # the value for remote_tmpdir should be the original one
    res = runner.invoke(cli.app, ['get', 'batch/remote_tmpdir'], catch_exceptions=False)
    assert res.exit_code == 0
    assert res.stdout.strip() == orig_remote_tmpdir
