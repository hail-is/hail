from collections import namedtuple
import os
import sys
import re
import warnings

from typing import Optional, Tuple, Annotated as Ann
from rich import print

import typer
from typer import Argument as Arg


app = typer.Typer(
    name='config',
    no_args_is_help=True,
    help='Manage Hail configuration.',
    pretty_exceptions_show_locals=False,
)


ConfigVariable = namedtuple('ConfigVariable', ['name', 'help_msg', 'validation'])


def config_variables():
    from hailtop.aiotools.router_fs import RouterAsyncFS  # pylint: disable=import-outside-toplevel
    from hailtop.batch_client.parse import CPU_REGEXPAT, MEMORY_REGEXPAT  # pylint: disable=import-outside-toplevel

    variables = [
        ConfigVariable(
            name='domain',
            help_msg='Domain of the Batch service',
            validation=(lambda x: re.fullmatch(r'.+\..+', x) is not None, 'should be valid domain'),
        ),
        ConfigVariable(
            name='gcs_requester_pays/project',
            help_msg='Project when using requester pays buckets in GCS',
            validation=(lambda x: re.fullmatch(r'[^:/\s]+', x) is not None, 'should be valid GCS project name'),
        ),
        ConfigVariable(
            name='gcs_requester_pays/buckets',
            help_msg='Allowed buckets when using requester pays in GCS',
            validation=(lambda x: re.fullmatch(r'[^:/\s]+(,[^:/\s]+)*', x) is not None, 'should be comma separated list of bucket names'),
        ),
        ConfigVariable(
            name='batch/bucket',
            help_msg='Deprecated - Name of GCS bucket to use as a temporary scratch directory',
            validation=(lambda x: re.fullmatch(r'[^:/\s]+', x) is not None, 'should be valid Google Bucket identifier, with no gs:// prefix'),
        ),
        ConfigVariable(
            name='batch/remote_tmpdir',
            help_msg='Cloud storage URI to use as a temporary scratch directory',
            validation=(RouterAsyncFS.valid_url, 'should be valid cloud storage URI such as gs://my-bucket/batch-tmp/'),
        ),
        ConfigVariable(
            name='batch/regions',
            help_msg='Comma-separated list of regions to run jobs in',
            validation=(lambda x: re.fullmatch(r'[^\s]+(,[^\s]+)*', x) is not None, 'should be comma separated list of regions'),
        ),
        ConfigVariable(
            name='batch/billing_project',
            help_msg='Batch billing project',
            validation=(lambda x: re.fullmatch(r'[^:/\s]+', x) is not None, 'should be valid Batch billing project name'),
        ),
        ConfigVariable(
            name='batch/backend',
            help_msg='Backend to use. One of local or service.',
            validation=(lambda x: x in ('local', 'service'), 'should be one of "local" or "service"'),
        ),
        ConfigVariable(
            name='query/backend',
            help_msg='Backend to use for Hail Query. One of spark, local, batch.',
            validation=(lambda x: x in ('local', 'spark', 'batch'), 'should be one of "local", "spark", or "batch"'),
        ),
        ConfigVariable(
            name='query/jar_url',
            help_msg='Cloud storage URI to a Query JAR',
            validation=(RouterAsyncFS.valid_url, 'should be valid cloud storage URI such as gs://my-bucket/jars/sha.jar')
        ),
        ConfigVariable(
            name='query/batch_driver_cores',
            help_msg='Cores specification for the query driver',
            validation=(lambda x: re.fullmatch(CPU_REGEXPAT, x) is not None, 'should be an integer which is a power of two from 1 to 16 inclusive'),
        ),
        ConfigVariable(
            name='query/batch_driver_memory',
            help_msg='Memory specification for the query driver',
            validation=(lambda x: re.fullmatch(MEMORY_REGEXPAT, x) is not None or x in ('standard', 'lowmem', 'highmem'),
                        'should be a valid string specifying memory "[+]?((?:[0-9]*[.])?[0-9]+)([KMGTP][i]?)?B?" or one of standard, lowmem, highmem'),
        ),
        ConfigVariable(
            name='query/batch_worker_cores',
            help_msg='Cores specification for the query worker',
            validation=(lambda x: re.fullmatch(CPU_REGEXPAT, x) is not None, 'should be an integer which is a power of two from 1 to 16 inclusive'),
        ),
        ConfigVariable(
            name='query/batch_worker_memory',
            help_msg='Memory specification for the query worker',
            validation=(lambda x: re.fullmatch(MEMORY_REGEXPAT, x) is not None or x in ('standard', 'lowmem', 'highmem'),
                        'should be a valid string specifying memory "[+]?((?:[0-9]*[.])?[0-9]+)([KMGTP][i]?)?B?" or one of standard, lowmem, highmem'),
        ),
        ConfigVariable(
            name='query/name_prefix',
            help_msg='Name used when displaying query progress in a progress bar',
            validation=(lambda x: re.fullmatch(r'[^\s]+', x) is not None, 'should be single word without spaces'),
        ),
        ConfigVariable(
            name='query/disable_progress_bar',
            help_msg='Disable the progress bar with a value of 1. Enable the progress bar with a value of 0',
            validation=(lambda x: x in ('0', '1'), 'should be a value of 0 or 1'),
        ),
    ]

    return variables


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
parameter, for example: "domain".

A parameter with more than one slash is invalid, for example:
"batch/billing/project".
'''.lstrip(
            '\n'
        ),
        file=sys.stderr,
    )
    sys.exit(1)


def complete_config_variable(incomplete: str):
    for var in config_variables():
        if var.name.startswith(incomplete):
            yield (var.name, var.help_msg)


@app.command()
def set(parameter: Ann[str, Arg(help="Configuration variable to set", autocompletion=complete_config_variable)], value: str):
    '''Set a Hail configuration parameter.'''
    from hailtop.config import get_user_config, get_user_config_path  # pylint: disable=import-outside-toplevel

    config = get_user_config()
    config_file = get_user_config_path()
    section, key, path = get_section_key_path(parameter)

    validations = {config.name: config.validation for config in config_variables()}

    validation_func, msg = validations.get(parameter, (lambda _: True, ''))  # type: ignore
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


def get_config_variable(incomplete: str):
    from hailtop.config import get_user_config  # pylint: disable=import-outside-toplevel

    config = get_user_config()

    elements = []
    for section_name, section in config.items():
        for item_name, value in section.items():
            if section_name == 'global':
                path = item_name
            else:
                path = f'{section_name}/{item_name}'
            elements.append((path, value))

    config_items = {config.name: config.help_msg for config in config_variables()}

    for name, _ in elements:
        if name.startswith(incomplete):
            help_msg = config_items.get(name)
            yield (name, help_msg)


@app.command()
def unset(parameter: Ann[str, Arg(help="Configuration variable to unset", autocompletion=get_config_variable)]):
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
def get(parameter: Ann[str, Arg(help="Configuration variable to get", autocompletion=get_config_variable)]):
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
