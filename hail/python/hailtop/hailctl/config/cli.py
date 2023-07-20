import os
import sys
import re
import warnings

from typing import Optional, Tuple, Annotated as Ann

import typer
from typer import Argument as Arg


app = typer.Typer(
    name='config',
    no_args_is_help=True,
    help='Manage Hail configuration.',
    pretty_exceptions_show_locals=False,
)


def get_section_key_path(parameter: str) -> Tuple[str, str, Tuple[str, ...]]:
    path = parameter.split('/')
    if len(path) == 1:
        return 'global', path[0], tuple(path)
    if len(path) == 2:
        return path[0], path[1], tuple(path)
    print(
        '''
Parameters must contain at most one slash separating the configuration section
from the configuration parameter, for example: "batch/billing_project".

Parameters may also have no slashes, indicating the parameter is a global
parameter, for example: "email".

A parameter with more than one slash is invalid, for example:
"batch/billing/project".
'''.lstrip(
            '\n'
        ),
        file=sys.stderr,
    )
    sys.exit(1)


@app.command()
def set(parameter: str, value: str):
    '''Set a Hail configuration parameter.'''
    from hailtop.aiotools.router_fs import RouterAsyncFS  # pylint: disable=import-outside-toplevel
    from hailtop.config import get_user_config, get_user_config_path  # pylint: disable=import-outside-toplevel

    config = get_user_config()
    config_file = get_user_config_path()
    section, key, path = get_section_key_path(parameter)

    validations = {
        ('batch', 'bucket'): (
            lambda x: re.fullmatch(r'[^:/\s]+', x) is not None,
            'should be valid Google Bucket identifier, with no gs:// prefix',
        ),
        ('batch', 'remote_tmpdir'): (
            RouterAsyncFS.valid_url,
            'should be valid cloud storage URI such as gs://my-bucket/batch-tmp/',
        ),
        ('email',): (lambda x: re.fullmatch(r'.+@.+', x) is not None, 'should be valid email address'),
    }

    validation_func, msg = validations.get(path, (lambda _: True, ''))  # type: ignore
    if not validation_func(value):
        print(f"Error: bad value {value!r} for parameter {parameter!r} {msg}", file=sys.stderr)
        sys.exit(1)

    if path == ('batch', 'bucket'):
        warnings.warn("'batch/bucket' has been deprecated. Use 'batch/remote_tmpdir' instead.")

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


@app.command()
def unset(parameter: str):
    '''Unset a Hail configuration parameter (restore to default behavior).'''
    from hailtop.config import get_user_config, get_user_config_path  # pylint: disable=import-outside-toplevel

    config = get_user_config()
    config_file = get_user_config_path()
    section, key, _ = get_section_key_path(parameter)
    if section in config and key in config[section]:
        del config[section][key]
        with open(config_file, 'w', encoding='utf-8') as f:
            config.write(f)


@app.command()
def get(parameter: str):
    '''Get the value of a Hail configuration parameter.'''
    from hailtop.config import get_user_config  # pylint: disable=import-outside-toplevel

    config = get_user_config()
    section, key, _ = get_section_key_path(parameter)
    if section in config and key in config[section]:
        print(config[section][key])


@app.command(name='config-location')
def config_location():
    '''Print the location of the config file.'''
    from hailtop.config import get_user_config_path  # pylint: disable=import-outside-toplevel

    print(get_user_config_path())


@app.command()
def list(section: Ann[Optional[str], Arg(show_default='all sections')] = None):
    '''Lists every config variable in the section.'''
    from hailtop.config import get_user_config  # pylint: disable=import-outside-toplevel

    config = get_user_config()
    if section:
        for key, value in config.items(section):
            print(f'{key}={value}')
    else:
        for sname, items in config.items():
            for key, value in items.items():
                print(f'{sname}/{key}={value}')
