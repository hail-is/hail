import sys
import re
import click

from hailtop.config import get_user_config, get_user_config_path

from ..hailctl import hailctl


validations = {
    ('batch', 'bucket'): (lambda x: re.fullmatch(r'[^:/\s]+', x) is not None,
                          'should be valid Google Bucket identifier, with no gs:// prefix'),
    ('global', 'email'): (lambda x: re.fullmatch(r'.+@.+', x) is not None, 'should be valid email address')
}


def split_parameter(parameter):
    path = parameter.split('/')
    if len(path) == 1:
        section = 'global'
        key = path[0]
    elif len(path) == 2:
        section = path[0]
        key = path[1]
    else:
        print('''
Paramters must contain at most one slash separating the configuration
section from the configuration parameter, for example:
"batch/billing_project".

Parameters may also have no slashes, indicating the parameter is a
global parameter, for example: "email".

A parameter with more than one slash is invalid, for example:
"batch/billing/project".
'''.lstrip('\n'), file=sys.stderr)
        sys.exit(1)
    return section, key


@hailctl.group()
def config():
    pass


@config.command(
    help="Set a Hail configuration parameter.")
@click.argument('parameter')
@click.argument('value')
def set(parameter, value):
    section, key = split_parameter(parameter)
    validation_func, msg = validations.get((section, key), (lambda x: True, ''))
    if not validation_func(value):
        print(f"Error: bad value {value!r} for parameter {key!r} {msg}", file=sys.stderr)
        sys.exit(1)
    config = get_user_config()
    if section not in config:
        config[section] = dict()
    config[section][key] = value

    config_file = get_user_config_path()
    with open(config_file, 'w') as f:
        config.write(f)


@config.command(
    help="Unset a Hail configuration parameter (restore to default behavior).")
@click.argument('parameter')
def unset(parameter):
    section, key = split_parameter(parameter)
    config = get_user_config()
    if section in config and key in config[section]:
        del config[section][key]
        config_file = get_user_config_path()
        with open(config_file, 'w') as f:
            config.write(f)


@config.command(
    help="Get the value of a Hail configuration parameter.")
@click.argument('parameter')
def get(parameter):
    section, key = split_parameter(parameter)
    config = get_user_config()
    if section in config and key in config[section]:
        print(config[section][key])


@config.command(name='list',
                help="List every config variable in the section. [default: (all sections)]")
@click.argument('section', required=False)
def config_list(section):
    config = get_user_config()
    if section:
        for key, value in config.items(section):
            print(f'{key}={value}')
    else:
        for sname, items in config.items():
            for key, value in items.items():
                print(f'{sname}/{key}={value}')


@config.command(
    help="Print the location of the config file")
def config_location():
    config_file = get_user_config_path()
    print(config_file)
