from typing import Optional, Union, TypeVar, Tuple, List
import os
import re
import configparser
import warnings
from dataclasses import dataclass

from pathlib import Path

user_config = None


def xdg_config_home() -> Path:
    value = os.environ.get('XDG_CONFIG_HOME')
    if value is None:
        return Path(Path.home(), ".config")
    return Path(value)


def get_user_config_path() -> Path:
    return Path(xdg_config_home(), 'hail', 'config.ini')


def get_user_config() -> configparser.ConfigParser:
    global user_config
    if user_config is None:
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
                'run `hailctl config set batch/remote_tmpdir REMOTE_TEMPTER`.')
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


def get_gcs_requester_pays_configuration(
        caller_name: str,
        *,
        gcs_requester_pays_configuration: Optional[Union[str, Tuple[str, List[str]]]] = None,
) -> Optional[Union[str, Tuple[str, List[str]]]]:
    if gcs_requester_pays_configuration:
        return gcs_requester_pays_configuration

    project = configuration_of('gcs_requester_pays', 'project', None, None)
    buckets = configuration_of('gcs_requester_pays', 'buckets', None, None)

    spark_conf = get_spark_conf_gcs_requester_pays_configuration()

    if spark_conf and (project or buckets):
        warnings.warn(
            f'You have specified the GCS requester pays configuration in both your '
            f'spark-defaults.conf ({spark_conf.conf_path}) and either an explicit argument or '
            f'through `hailctl config`. For GCS requester pays configuration, Hail '
            'first checks explicit arguments, then `hailctl config`, then '
            'spark-defaults.conf.'
        )

    if project is not None:
        if buckets is not None:
            return (project, buckets.split(','))
        return project

    if spark_conf is None:
        return None
    warnings.warn(
        'Reading spark-defaults.conf to determine GCS requester pays '
        'configuration. This is deprecated. Please use `hailctl config '
        'set gcs_requeseter_pays/project` and `hailctl config set '
        'gcs_requester_pays/buckets`.'
    )

    # https://github.com/GoogleCloudDataproc/hadoop-connectors/blob/master/gcs/CONFIGURATION.md#cloud-storage-requester-pays-feature-configuration
    if spark_conf.mode is None:
        if spark_conf.project or spark_conf.buckets:
            warnings.warn(
                f'When reading GCS requester pays configuration from spark-defaults.conf '
                f'({spark_conf.conf_path}), no mode is set, so requester pays '
                f'will be disabled.'
           )
        return None

    if spark_conf.mode == 'DISABLED':
        return None

    if spark_conf.project is None:
        raise ValueError(
            f'When reading GCS requester pays configuration from spark-defaults.conf '
            f'({spark_conf.conf_path}), a project must be set if a mode other than '
            f'DISABLED is set.'
        )

    if spark_conf.mode == 'CUSTOM':
        if spark_conf.buckets is None:
            raise ValueError(
                f'When reading GCS requester pays configuration from '
                f'spark-defaults.conf ({spark_conf.conf_path}) with mode CUSTOM, '
                f'buckets must be set.'
            )
        return (spark_conf.project, spark_conf.buckets)

    if spark_conf.mode not in ('ENABLED', 'AUTO'):
        raise ValueError(
            f'When reading GCS requester pays configuration from '
            f'spark-defaults.conf ({spark_conf.conf_path}) an unknown mode was '
            f'found: {spark_conf.mode}. Expected ENABLED, AUTO, CUSTOM, or '
            f'DISABLED.'
        )

    if spark_conf.mode == 'ENABLED':
        warnings.warn(
            f'When reading GCS requester pays configuration from '
            f'spark-defaults.conf ({spark_conf.conf_path}) Hail treats the mode '
            f'ENABLED as AUTO.'
        )

    if spark_conf.buckets is not None:
        warnings.warn(
            f'When reading GCS requester pays configuration from '
            f'spark-defaults.conf ({spark_conf.conf_path}) with mode '
            f'{spark_conf.mode}, found buckets: {spark_conf.buckets}.'
            f'The buckets are ignored in this mode.'
        )

    return spark_conf.project


@dataclass
class SparkConfGcsRequseterPaysConfiguration:
    mode: Optional[str]
    project: Optional[str]
    buckets: Optional[List[str]]
    conf_path: str


def spark_conf_path() -> Optional[str]:
    try:
        from pyspark.find_spark_home import _find_spark_home
    except ImportError:
        return None
    return os.path.join(_find_spark_home(), 'conf', 'spark-defaults.conf')


def get_spark_conf_gcs_requester_pays_configuration() -> Optional[SparkConfGcsRequseterPaysConfiguration]:
    mode: Optional[str] = None
    project: Optional[str] = None
    buckets: Optional[List[str]] = None
    path = spark_conf_path()
    if path is not None and os.path.exists(path):
        with open(path) as f:
            for line in f.readlines():
                setting = line.rstrip('\n')
                maybe_var_and_val = setting.split(' ')
                if len(maybe_var_and_val) != 2:
                    raise ValueError(f'Found spark-defaults.conf file line with more than one space: {line}')
                var, val = maybe_var_and_val
                if var == 'spark.hadoop.fs.gs.requester.pays.mode':
                    mode = val
                if var == 'spark.hadoop.fs.gs.requester.pays.project.id':
                    project = val
                if var == 'spark.hadoop.fs.gs.requester.pays.buckets':
                    buckets = val.split(',')
        if mode or project or buckets:
            return SparkConfGcsRequseterPaysConfiguration(mode, project, buckets, path)
    return None
