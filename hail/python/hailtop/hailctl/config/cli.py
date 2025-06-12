import os
import sys
from collections import defaultdict
from typing import Annotated as Ann
from typing import Generator, Optional, Tuple

import typer
from rich.console import Console
from typer import Argument as Arg

from hailtop.config.variables import ConfigVariable

from .config_variables import config_variables

app = typer.Typer(
    name='config',
    no_args_is_help=True,
    help='Manage Hail configuration.',
    pretty_exceptions_show_locals=False,
)

profile_app = typer.Typer(
    name='profile',
    no_args_is_help=True,
    help='Manage Hail configuration profiles.',
    pretty_exceptions_show_locals=False,
)

app.add_typer(profile_app)

outc = Console(soft_wrap=True)
errc = Console(stderr=True, soft_wrap=True)


def get_section_key_path(parameter: str) -> Tuple[str, str, Tuple[str, ...]]:
    path = parameter.split('/')
    if len(path) == 1:
        return 'global', path[0], tuple(path)
    if len(path) == 2:
        return path[0], path[1], tuple(path)
    errc.print(
        """
Parameters must contain at most one slash separating the configuration section
from the configuration parameter, for example: "batch/billing_project".

Parameters may also have no slashes, indicating the parameter is a global
parameter, for example: "domain".

A parameter with more than one slash is invalid, for example:
"batch/billing/project".
""".lstrip('\n'),
    )
    sys.exit(1)


def complete_config_variable(incomplete: str):
    for var, var_info in config_variables().items():
        if var.value.startswith(incomplete):
            yield (var.value, var_info.help_msg)


@app.command()
def set(
    parameter: Ann[ConfigVariable, Arg(help="Configuration variable to set", autocompletion=complete_config_variable)],
    value: str,
):
    """Set a Hail configuration parameter."""
    from hailtop.config import (  # pylint: disable=import-outside-toplevel
        get_config_from_file,
        get_config_profile_name,
        get_user_config_path_by_profile_name,
    )

    if parameter not in config_variables():
        errc.print(f"Error: unknown parameter {parameter!r}")
        sys.exit(1)

    section, key, _ = get_section_key_path(parameter.value)

    config_variable_info = config_variables()[parameter]
    validation_func, error_msg = config_variable_info.validation

    if not validation_func(value):
        errc.print(f"Error: bad value {value!r} for parameter {parameter!r} {error_msg}")
        sys.exit(1)

    profile_name = get_config_profile_name()

    if parameter != ConfigVariable.PROFILE:
        config_file = get_user_config_path_by_profile_name(profile_name=profile_name)
    else:
        config_file = get_user_config_path_by_profile_name(profile_name=None)

    config, _ = get_config_from_file(config_file)

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


@app.command()
def unset(parameter: Ann[str, Arg(help="Configuration variable to unset", autocompletion=get_config_variable)]):
    """Unset a Hail configuration parameter (restore to default behavior)."""
    from hailtop.config import get_user_config, get_user_config_path  # pylint: disable=import-outside-toplevel

    config = get_user_config()
    config_file = get_user_config_path()
    section, key, _ = get_section_key_path(parameter)
    if section in config and key in config[section]:
        del config[section][key]
        with open(config_file, 'w', encoding='utf-8') as f:
            config.write(f)
    else:
        errc.print(f"WARNING: Unknown parameter {parameter!r}")


@app.command()
def get(parameter: Ann[str, Arg(help="Configuration variable to get", autocompletion=get_config_variable)]):
    """Get the value of a Hail configuration parameter."""
    from hailtop.config import get_user_config  # pylint: disable=import-outside-toplevel

    config = get_user_config()
    section, key, _ = get_section_key_path(parameter)
    if section in config and key in config[section]:
        outc.print(config[section][key])


@app.command(name='config-location')
def config_location():
    """Print the location of the config file."""
    from hailtop.config import (  # pylint: disable=import-outside-toplevel
        get_config_profile_name,
        get_user_config_path,
        get_user_config_path_by_profile_name,
    )

    outc.print(f'Default settings: {get_user_config_path()}')

    profile_name = get_config_profile_name()
    if profile_name is not None:
        profile_path = get_user_config_path_by_profile_name(profile_name=profile_name)
        outc.print(f'Overrode default settings with profile "{profile_name}": {profile_path}')


@app.command(name='list')
def list_config(section: Ann[Optional[str], Arg(show_default='all sections')] = None):
    """Lists every config variable in the section."""
    from hailtop.config import (  # pylint: disable=import-outside-toplevel
        get_user_config_with_profile_overrides_and_source,
    )

    _, source = get_user_config_with_profile_overrides_and_source()

    grouped_source = defaultdict(list)
    for (_section, option), (value, path) in source.items():
        if section is None or _section == section:
            grouped_source[path].append(((_section, option), value))

    output = []
    for path, values in grouped_source.items():
        output.append(f'Config settings from {path}:\n')
        for (_section, option), value in values:
            output.append(
                f'[cyan]{_section}[/cyan]/[yellow]{option}[/yellow]=[bright_magenta]{value}[/bright_magenta]\n'
            )
        output.append('\n')

    if output:
        outc.print(''.join(output).rstrip('\n'))


def _list_profiles():
    from hailtop.config import get_hail_config_path  # pylint: disable=import-outside-toplevel

    profiles = ['default']
    for file in os.listdir(get_hail_config_path()):
        if file.endswith('.ini') and file != 'config.ini':
            profile_name = file[:-4]
            profiles.append(profile_name)

    profiles.sort()

    return profiles


@profile_app.command(name='list')
def list_profiles():
    """List the available Hail configuration profiles."""
    from hailtop.config import get_config_profile_name  # pylint: disable=import-outside-toplevel

    profiles = _list_profiles()
    current_profile = get_config_profile_name() or 'default'

    for profile in profiles:
        if profile == current_profile:
            outc.print(f'* [green]{profile}[/green]')
        else:
            outc.print(f'  {profile}')


def _get_profile(incomplete: str) -> Generator[str, None, None]:
    profiles = _list_profiles()
    for profile in profiles:
        if profile.startswith(incomplete):
            yield profile


@profile_app.command(name='load')
def load_profile(
    profile_name: Ann[str, Arg(help='Name of configuration profile to load.', autocompletion=_get_profile)] = 'default',
):
    """Load a Hail configuration profile."""
    from hailtop.config import get_user_config_path_by_profile_name  # pylint: disable=import-outside-toplevel

    if profile_name == 'default':
        config_file = get_user_config_path_by_profile_name(profile_name=None)
    else:
        config_file = get_user_config_path_by_profile_name(profile_name=profile_name)

    if not os.path.exists(config_file):
        errc.print(
            f"Error: profile '{profile_name}' does not exist. Use `hailctl config profile create {profile_name}` to create it.",
        )
        sys.exit(1)

    set(ConfigVariable.PROFILE, profile_name)

    outc.print(f'Loaded profile {profile_name} with settings:\n')
    list_config()


@profile_app.command(name='create')
def create_profile(profile_name: Ann[str, Arg(help='Name of configuration profile to create.')]):
    """Create a new Hail configuration profile."""
    from hailtop.config import get_user_config_path_by_profile_name  # pylint: disable=import-outside-toplevel

    profile_config_file = get_user_config_path_by_profile_name(profile_name=profile_name)

    if profile_config_file.is_file():
        errc.print(f'Error: profile {profile_name} already exists!')
        sys.exit(1)

    try:
        profile_config_file.touch()
    except FileNotFoundError:
        os.makedirs(profile_config_file.parent, exist_ok=True)
        profile_config_file.touch()

    outc.print(
        f'Created profile "{profile_name}". You can load the profile with `hailctl config profile load {profile_name}`.'
    )


@profile_app.command(name='delete')
def delete_profile(
    profile_name: Ann[str, Arg(help='Name of configuration profile to delete.', autocompletion=_get_profile)],
):
    """Delete a Hail configuration profile."""
    from hailtop.config import (  # pylint: disable=import-outside-toplevel
        get_config_profile_name,
        get_user_config_path_by_profile_name,
    )

    if profile_name == 'default':
        errc.print('Cannot delete the "default" profile.')
        sys.exit(1)

    current_profile = get_config_profile_name()
    if current_profile == profile_name:
        errc.print(
            'Cannot delete a profile that is currently being used. Use `hailctl config profile list` to see available profiles. Load a different environment with `hailctl config profile load <profile_name>`.'
        )
        sys.exit(1)

    profile_config_file = get_user_config_path_by_profile_name(profile_name=profile_name)
    try:
        os.remove(profile_config_file)
    except FileNotFoundError:
        errc.print(f'Unknown profile "{profile_name}".')
        sys.exit(1)

    outc.print(f'Deleted profile "{profile_name}".')
