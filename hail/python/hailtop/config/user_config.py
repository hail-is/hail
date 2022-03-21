from typing import Optional
import os
import configparser
import warnings

from pathlib import Path

user_config = None


def xdg_config_home() -> Path:
    value = os.environ.get('XDG_CONFIG_HOME')
    if value is None:
        return Path(Path.home(), ".config")
    return Path(value)


def get_user_local_cache_dir(ensure_exists: bool = False) -> Path:
    cache_dir = Path(xdg_config_home(), 'hail', 'cache')
    if ensure_exists:
        os.makedirs(cache_dir, exist_ok=True)
    return cache_dir


def get_user_config_path() -> Path:
    return Path(xdg_config_home(), 'hail', 'config.ini')


def get_user_config() -> configparser.ConfigParser:
    global user_config
    if user_config is None:
        user_config = configparser.ConfigParser()
        config_file = get_user_config_path()
        os.makedirs(config_file.parent, exist_ok=True)
        # in older versions, the config file was accidentally named
        # config.yaml, if the new config does not exist, and the old
        # one does, silently rename it
        old_path = config_file.with_name('config.yaml')
        if old_path.exists() and not config_file.exists():
            old_path.rename(config_file)
        else:
            config_file.touch(exist_ok=True)
        user_config.read(config_file)
    return user_config


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
        remote_tmpdir = user_config.get('batch', 'remote_tmpdir', fallback=None)

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
        schemes = {'gs', 'hail-az'}
        found_scheme = any(remote_tmpdir.startswith(f'{scheme}://') for scheme in schemes)
        if not found_scheme:
            raise ValueError(
                f'remote_tmpdir must be a storage uri path like gs://bucket/folder. Possible schemes include {schemes}')
    if remote_tmpdir[-1] != '/':
        remote_tmpdir += '/'
    return remote_tmpdir
