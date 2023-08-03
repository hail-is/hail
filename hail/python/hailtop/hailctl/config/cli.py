import os
import sys
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
    from hailtop.config import config_variables  # pylint: disable=import-outside-toplevel

    for var, var_info in config_variables.items():
        if var.value.startswith(incomplete):
            yield (var.value, var_info.help_msg)


@app.command()
def set(parameter: Ann[str, Arg(help="Configuration variable to set", autocompletion=complete_config_variable)], value: str):
    '''Set a Hail configuration parameter.'''
    from hailtop.config import config_variables, get_user_config, get_user_config_path  # pylint: disable=import-outside-toplevel

    config = get_user_config()
    config_file = get_user_config_path()

    if parameter not in config_variables:
        print(f"Error: unknown parameter {parameter!r}", file=sys.stderr)
        sys.exit(1)

    section, key, path = get_section_key_path(parameter)

    validations = {var.name: var_info.validation for var, var_info in config_variables.items()}

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
    from hailtop.config import config_variables, get_user_config  # pylint: disable=import-outside-toplevel

    config = get_user_config()

    elements = []
    for section_name, section in config.items():
        for item_name, value in section.items():
            if section_name == 'global':
                path = item_name
            else:
                path = f'{section_name}/{item_name}'
            elements.append((path, value))

    config_items = {var.name: var_info.help_msg for var, var_info in config_variables.items()}

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
