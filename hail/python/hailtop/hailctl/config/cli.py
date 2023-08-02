import asyncio
import os
import sys

from typing import Optional, Tuple, Annotated as Ann
from rich import print

from rich.prompt import Prompt

import typer
from typer import Abort, Argument as Arg, Exit, Option as Opt


from .initialize import async_basic_initialize

from hailtop.config.variables import ConfigVariable
from .config_variables import config_variables


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
    for var, var_info in config_variables().items():
        if var.value.startswith(incomplete):
            yield (var.value, var_info.help_msg)


@app.command(help='Set the value of a configuration parameter.')
def set(parameter: Ann[ConfigVariable, Arg(help="Configuration variable to set", autocompletion=complete_config_variable)], value: str):
    '''Set a Hail configuration parameter.'''
    from hailtop.config import get_user_config, get_user_config_path  # pylint: disable=import-outside-toplevel

    if parameter not in config_variables():
        print(f"Error: unknown parameter {parameter!r}", file=sys.stderr)
        sys.exit(1)

    section, key, _ = get_section_key_path(parameter.value)

    config_variable_info = config_variables()[parameter]
    validation_func, error_msg  = config_variable_info.validation

    if not validation_func(value):
        print(f"Error: bad value {value!r} for parameter {parameter!r} {error_msg}", file=sys.stderr)
        sys.exit(1)

    config = get_user_config()
    config_file = get_user_config_path()

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

    config_items = {var.name: var_info.help_msg for var, var_info in config_variables().items()}

    for name, _ in elements:
        if name.startswith(incomplete):
            help_msg = config_items.get(name)
            yield (name, help_msg)


# @app.command()
# def set(parameter: str, value: str):
#     _set(parameter, value)


@app.command(help='Unset the value of a configuration parameter.')
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
    else:
        print(f"WARNING: Unknown parameter {parameter!r}", file=sys.stderr)


@app.command(help='Get the value for a configuration parameter.')
def get(parameter: Ann[str, Arg(help="Configuration variable to get", autocompletion=get_config_variable)]):
    '''Get the value of a Hail configuration parameter.'''
    from hailtop.config import get_user_config  # pylint: disable=import-outside-toplevel

    config = get_user_config()
    section, key, _ = get_section_key_path(parameter)
    if section in config and key in config[section]:
        print(config[section][key])


@app.command(name='config-location', help='Show the path to the configuration file.')
def config_location():
    '''Print the location of the config file.'''
    from hailtop.config import get_user_config_path  # pylint: disable=import-outside-toplevel

    print(get_user_config_path())


@app.command(help='List all configuration variables.')
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


@app.command(help='Clear an existing Hail environment.')
def clear():
    from hailtop.config import get_user_config_path  # pylint: disable=import-outside-toplevel

    config_file = get_user_config_path()
    if os.path.isfile(config_file):
        os.remove(config_file)


@app.command('init', help='Initialize a Hail environment.')
def initialize(
        overwrite: Ann[bool, Opt('--overwrite', '-o',
                                 help='Destroy the existing configuration before setting new variables.')] = False,
        verbose: Ann[bool, Opt('--verbose', '-v', help='Print gcloud commands being executed')] = False
):
    if overwrite:
        clear()
    asyncio.get_event_loop().run_until_complete(async_basic_initialize(verbose=verbose))
