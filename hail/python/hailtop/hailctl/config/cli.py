import os
import sys
import re
import warnings

from typing import Optional, Tuple, Annotated as Ann
from rich import print

import typer
from typer import Argument as Arg

from hailtop.batch_client.parse import CPU_REGEXPAT, MEMORY_REGEXPAT

app = typer.Typer(
    name='config',
    no_args_is_help=True,
    help='Manage Hail configuration.',
    pretty_exceptions_show_locals=False,
)


_config_variables = None


def config_variables():
    from hailtop.aiotools.router_fs import RouterAsyncFS  # pylint: disable=import-outside-toplevel

    global _config_variables

    if _config_variables is not None:
        return _config_variables

    variables = [
        (
            'domain',
            'Domain of the Batch service',
            (lambda x: re.fullmatch(r'.+\..+', x) is not None, 'should be valid domain'),
        ),
        (
            'gcs_requester_pays/project',
            'Project when using requester pays buckets in GCS',
            (lambda x: re.fullmatch(r'[^:/\s]+', x) is not None, 'should be valid GCS project name'),
        ),
        (
            'gcs_requester_pays/buckets',
            'Allowed buckets when using requester pays in GCS',
            (lambda x: re.fullmatch(r'[^:/\s]+(,[^:/\s]+)*', x) is not None, 'should be comma separated list of bucket names'),
        ),
        (
            'batch/bucket',
            'Deprecated - Name of GCS bucket to use as a temporary scratch directory',
            (lambda x: re.fullmatch(r'[^:/\s]+', x) is not None, 'should be valid Google Bucket identifier, with no gs:// prefix'),
        ),
        (
            'batch/remote_tmpdir',
            'Cloud storage URI to use as a temporary scratch directory',
            (RouterAsyncFS.valid_url, 'should be valid cloud storage URI such as gs://my-bucket/batch-tmp/'),
        ),
        (
            'batch/regions',
            'Comma-separated list of regions to run jobs in',
            (lambda x: re.fullmatch(r'[^\s]+(,[^\s]+)*', x) is not None, 'should be comma separated list of regions'),
        ),
        (
            'batch/billing_project',
            'Batch billing project',
            (lambda x: re.fullmatch(r'[^:/\s]+', x) is not None, 'should be valid Batch billing project name'),
        ),
        (
            'batch/backend',
            'Backend to use. One of local or service.',
            (lambda x: x in ('local', 'service'), 'should be one of "local" or "service"'),
        ),
        (
            'query/backend',
            'Backend to use for Hail Query. One of spark, local, batch.',
            (lambda x: x in ('local', 'spark', 'batch'), 'should be one of "local", "spark", or "batch"'),
        ),
        (
            'query/jar_url',
            'Cloud storage URI to a Query JAR',
            (RouterAsyncFS.valid_url, 'should be valid cloud storage URI such as gs://my-bucket/jars/sha.jar')
        ),
        (
            'query/batch_driver_cores',
            'Cores specification for the query driver',
            (lambda x: re.fullmatch(CPU_REGEXPAT, x) is not None, 'should be an integer which is a power of two from 1 to 16 inclusive'),
        ),
        (
            'query/batch_driver_memory',
            'Memory specification for the query driver',
            (lambda x: re.fullmatch(MEMORY_REGEXPAT, x) is not None or x in ('standard', 'lowmem', 'highmem'),
             'should be a valid string specifying memory "[+]?((?:[0-9]*[.])?[0-9]+)([KMGTP][i]?)?B?" or one of standard, lowmem, highmem'),
        ),
        (
            'query/batch_worker_cores',
            'Cores specification for the query worker',
            (lambda x: re.fullmatch(CPU_REGEXPAT, x) is not None, 'should be an integer which is a power of two from 1 to 16 inclusive'),
        ),
        (
            'query/batch_worker_memory',
            'Memory specification for the query worker',
            (lambda x: re.fullmatch(MEMORY_REGEXPAT, x) is not None or x in ('standard', 'lowmem', 'highmem'),
             'should be a valid string specifying memory "[+]?((?:[0-9]*[.])?[0-9]+)([KMGTP][i]?)?B?" or one of standard, lowmem, highmem'),
        ),
        (
            'query/name_prefix',
            'Name used when displaying query progress in a progress bar',
            (lambda x: re.fullmatch(r'[^\s]+', x) is not None, 'should be single word without spaces'),
        ),
        (
            'query/disable_progress_bar',
            'Disable the progress bar with a value of 1. Enable the progress bar with a value of 0',
            (lambda x: x in ('0', '1'), 'should be a value of 0 or 1'),
        ),
    ]
    _config_variables = variables
    return _config_variables


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


def complete_config_variable(incomplete: str):
    for name, help_text, validation in config_variables():
        if name.startswith(incomplete):
            completion_item = (name, help_text)
            yield completion_item


@app.command()
def set(parameter: Ann[str, Arg(default=..., help="Configuration variable to set", autocompletion=complete_config_variable)], value: str):
    '''Set a Hail configuration parameter.'''
    from hailtop.config import get_user_config, get_user_config_path  # pylint: disable=import-outside-toplevel

    config = get_user_config()
    config_file = get_user_config_path()
    section, key, path = get_section_key_path(parameter)

    validations = {name: validation for name, _, validation in config_variables()}

    if parameter not in config_variables:
        sys.exit(1)

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


from rich.console import Console

console = Console(stderr=True)


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

    for name, current_value in elements:
        if name.startswith(incomplete):
            completion_item = (name, current_value)
            yield completion_item


@app.command()
def unset(parameter: Ann[str, Arg(default=..., help="Configuration variable to unset", autocompletion=get_config_variable)]):
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
def get(parameter: Ann[str, Arg(default=..., help="Configuration variable to get", autocompletion=get_config_variable)]):
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
