from collections import namedtuple
from enum import Enum
from typing import Optional, Union, TypeVar
import os
import re
import configparser
import warnings
from pathlib import Path

from hailtop.batch_client.parse import CPU_REGEXPAT, MEMORY_REGEXPAT
from hailtop.fs.router_fs import RouterAsyncFS

user_config = None


ConfigVariableInfo = namedtuple('ConfigVariable', ['help_msg', 'validation'])


class ConfigVariable(Enum):
    DOMAIN = 'domain'
    GCS_REQUESTER_PAYS_PROJECT = 'gcs_requester_pays/project',
    GCS_REQUESTER_PAYS_BUCKETS = 'gcs_requester_pays/buckets',
    BATCH_BUCKET = 'batch/bucket'
    BATCH_REMOTE_TMPDIR = 'batch/remote_tmpdir'
    BATCH_REGIONS = 'batch/regions',
    BATCH_BILLING_PROJECT = 'batch/billing_project'
    BATCH_BACKEND = 'batch/backend'
    QUERY_BACKEND = 'query/backend'
    QUERY_JAR_URL = 'query/jar_url'
    QUERY_BATCH_DRIVER_CORES = 'query/batch_driver_cores'
    QUERY_BATCH_WORKER_CORES = 'query/batch_worker_cores'
    QUERY_BATCH_DRIVER_MEMORY = 'query/batch_driver_memory'
    QUERY_BATCH_WORKER_MEMORY = 'query/batch_worker_memory'
    QUERY_NAME_PREFIX = 'query/name_prefix'
    QUERY_DISABLE_PROGRESS_BAR = 'query/disable_progress_bar'


config_variables = {
    ConfigVariable.DOMAIN: ConfigVariableInfo(
        help_msg='Domain of the Batch service',
        validation=(lambda x: re.fullmatch(r'.+\..+', x) is not None, 'should be valid domain'),
    ),
    ConfigVariable.GCS_REQUESTER_PAYS_PROJECT: ConfigVariableInfo(
        help_msg='Project when using requester pays buckets in GCS',
        validation=(lambda x: re.fullmatch(r'[^:/\s]+', x) is not None, 'should be valid GCS project name'),
    ),
    ConfigVariable.GCS_REQUESTER_PAYS_BUCKETS: ConfigVariableInfo(
        help_msg='Allowed buckets when using requester pays in GCS',
        validation=(
            lambda x: re.fullmatch(r'[^:/\s]+(,[^:/\s]+)*', x) is not None,
            'should be comma separated list of bucket names'),
    ),
    ConfigVariable.BATCH_BUCKET: ConfigVariableInfo(
        help_msg='Deprecated - Name of GCS bucket to use as a temporary scratch directory',
        validation=(lambda x: re.fullmatch(r'[^:/\s]+', x) is not None,
                    'should be valid Google Bucket identifier, with no gs:// prefix'),
    ),
    ConfigVariable.BATCH_REMOTE_TMPDIR: ConfigVariableInfo(
        help_msg='Cloud storage URI to use as a temporary scratch directory',
        validation=(RouterAsyncFS.valid_url, 'should be valid cloud storage URI such as gs://my-bucket/batch-tmp/'),
    ),
    ConfigVariable.BATCH_REGIONS: ConfigVariableInfo(
        help_msg='Comma-separated list of regions to run jobs in',
        validation=(
            lambda x: re.fullmatch(r'[^\s]+(,[^\s]+)*', x) is not None, 'should be comma separated list of regions'),
    ),
    ConfigVariable.BATCH_BILLING_PROJECT: ConfigVariableInfo(
        help_msg='Batch billing project',
        validation=(lambda x: re.fullmatch(r'[^:/\s]+', x) is not None, 'should be valid Batch billing project name'),
    ),
    ConfigVariable.BATCH_BACKEND: ConfigVariableInfo(
        help_msg='Backend to use. One of local or service.',
        validation=(lambda x: x in ('local', 'service'), 'should be one of "local" or "service"'),
    ),
    ConfigVariable.QUERY_BACKEND: ConfigVariableInfo(
        help_msg='Backend to use for Hail Query. One of spark, local, batch.',
        validation=(lambda x: x in ('local', 'spark', 'batch'), 'should be one of "local", "spark", or "batch"'),
    ),
    ConfigVariable.QUERY_JAR_URL: ConfigVariableInfo(
        help_msg='Cloud storage URI to a Query JAR',
        validation=(RouterAsyncFS.valid_url, 'should be valid cloud storage URI such as gs://my-bucket/jars/sha.jar')
    ),
    ConfigVariable.QUERY_BATCH_DRIVER_CORES: ConfigVariableInfo(
        help_msg='Cores specification for the query driver',
        validation=(lambda x: re.fullmatch(CPU_REGEXPAT, x) is not None,
                    'should be an integer which is a power of two from 1 to 16 inclusive'),
    ),
    ConfigVariable.QUERY_BATCH_WORKER_CORES: ConfigVariableInfo(
        help_msg='Cores specification for the query worker',
        validation=(lambda x: re.fullmatch(CPU_REGEXPAT, x) is not None,
                    'should be an integer which is a power of two from 1 to 16 inclusive'),
    ),
    ConfigVariable.QUERY_BATCH_DRIVER_MEMORY: ConfigVariableInfo(
        help_msg='Memory specification for the query driver',
        validation=(lambda x: re.fullmatch(MEMORY_REGEXPAT, x) is not None or x in ('standard', 'lowmem', 'highmem'),
                    'should be a valid string specifying memory "[+]?((?:[0-9]*[.])?[0-9]+)([KMGTP][i]?)?B?" or one of standard, lowmem, highmem'),
    ),
    ConfigVariable.QUERY_BATCH_WORKER_MEMORY: ConfigVariableInfo(
        help_msg='Memory specification for the query worker',
        validation=(lambda x: re.fullmatch(MEMORY_REGEXPAT, x) is not None or x in ('standard', 'lowmem', 'highmem'),
                    'should be a valid string specifying memory "[+]?((?:[0-9]*[.])?[0-9]+)([KMGTP][i]?)?B?" or one of standard, lowmem, highmem'),
    ),
    ConfigVariable.QUERY_NAME_PREFIX: ConfigVariableInfo(
        help_msg='Name used when displaying query progress in a progress bar',
        validation=(lambda x: re.fullmatch(r'[^\s]+', x) is not None, 'should be single word without spaces'),
    ),
    ConfigVariable.QUERY_DISABLE_PROGRESS_BAR: ConfigVariableInfo(
        help_msg='Disable the progress bar with a value of 1. Enable the progress bar with a value of 0',
        validation=(lambda x: x in ('0', '1'), 'should be a value of 0 or 1'),
    ),
}


def xdg_config_home() -> Path:
    value = os.environ.get('XDG_CONFIG_HOME')
    if value is None:
        return Path(Path.home(), ".config")
    return Path(value)


def get_user_config_path() -> Path:
    return Path(xdg_config_home(), 'hail', 'config.ini')


def get_user_config() -> configparser.ConfigParser:
    user_config = configparser.ConfigParser()
    config_file = get_user_config_path()
    # in older versions, the config file was accidentally named
    # config.yaml, if the new config does not exist, and the old
    # one does, silently rename it
    old_path = config_file.with_name('config.yaml')
    if old_path.exists() and not config_file.exists():
        old_path.rename(config_file)
    user_config.read(config_file)
    return user_config


VALID_SECTION_AND_OPTION_RE = re.compile('[a-z0-9_]+')
T = TypeVar('T')


def configuration_of(section: str,
                     option: str,
                     explicit_argument: Optional[T],
                     fallback: T,
                     *,
                     deprecated_envvar: Optional[str] = None) -> Union[str, T]:
    assert VALID_SECTION_AND_OPTION_RE.fullmatch(section), (section, option)
    assert VALID_SECTION_AND_OPTION_RE.fullmatch(option), (section, option)

    if section != 'global':
        path = f'{section}/{option}'
    else:
        path = option

    assert path in config_variables.keys()

    if explicit_argument is not None:
        return explicit_argument

    envvar = 'HAIL_' + section.upper() + '_' + option.upper()
    envval = os.environ.get(envvar, None)
    deprecated_envval = None if deprecated_envvar is None else os.environ.get(deprecated_envvar)
    if envval is not None:
        if deprecated_envval is not None:
            raise ValueError(f'Value for configuration variable {section}/{option} is ambiguous '
                             f'because both {envvar} and {deprecated_envvar} are set (respectively '
                             f'to: {envval} and {deprecated_envval}.')
        return envval
    if deprecated_envval is not None:
        warnings.warn(f'Use of deprecated envvar {deprecated_envvar} for configuration variable '
                      f'{section}/{option}. Please use {envvar} instead.')
        return deprecated_envval

    from_user_config = get_user_config().get(section, option, fallback=None)
    if from_user_config is not None:
        return from_user_config

    return fallback


def get_remote_tmpdir(caller_name: str,
                      *,
                      bucket: Optional[str] = None,
                      remote_tmpdir: Optional[str] = None,
                      user_config: Optional[configparser.ConfigParser] = None,
                      warnings_stacklevel: int = 2,
                      ) -> str:
    if user_config is None:
        user_config = get_user_config()

    if bucket is not None:
        warnings.warn(f'Use of deprecated argument \'bucket\' in {caller_name}(...). Specify \'remote_tmpdir\' as a keyword argument instead.',
                      stacklevel=warnings_stacklevel)

    if remote_tmpdir is not None and bucket is not None:
        raise ValueError(f'Cannot specify both \'remote_tmpdir\' and \'bucket\' in {caller_name}(...). Specify \'remote_tmpdir\' as a keyword argument instead.')

    if bucket is None and remote_tmpdir is None:
        remote_tmpdir = configuration_of('batch', 'remote_tmpdir', None, None)

    if remote_tmpdir is None:
        if bucket is None:
            bucket = user_config.get('batch', 'bucket', fallback=None)
            warnings.warn('Using deprecated configuration setting \'batch/bucket\'. Run `hailctl config set batch/remote_tmpdir` '
                          'to set the default for \'remote_tmpdir\' instead.',
                          stacklevel=warnings_stacklevel)
        if bucket is None:
            raise ValueError(
                f'Either the \'remote_tmpdir\' parameter of {caller_name}(...) must be set or you must '
                'run `hailctl config set batch/remote_tmpdir REMOTE_TMPDIR`.')
        if 'gs://' in bucket:
            raise ValueError(
                f'The bucket parameter to {caller_name}(...) and the `batch/bucket` hailctl config setting '
                'must both be bucket names, not paths. Use the remote_tmpdir parameter or batch/remote_tmpdir '
                'hailctl config setting instead to specify a path.')
        remote_tmpdir = f'gs://{bucket}/batch'
    else:
        schemes = {'gs', 'hail-az', 'https'}
        found_scheme = any(remote_tmpdir.startswith(f'{scheme}://') for scheme in schemes)
        if not found_scheme:
            raise ValueError(
                f'remote_tmpdir must be a storage uri path like gs://bucket/folder. Received: {remote_tmpdir}. Possible schemes include gs for GCP and https for Azure')
    if remote_tmpdir[-1] != '/':
        remote_tmpdir += '/'
    return remote_tmpdir
