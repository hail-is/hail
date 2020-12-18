import sys
import re
import click

from hailtop.config import get_user_config, get_user_config_path

from ..hailctl import hailctl


validations = {
    ('batch', 'bucket'): (lambda x: re.fullmatch(r'[^:/\s]+', x) is not None,
                          'should be valid Google Bucket identifier, with no gs:// prefix'),
    ('email',): (lambda x: re.fullmatch(r'.+@.+', x) is not None, 'should be valid email address')
}


@hailctl.group()
def config(args):
    pass


@config.command(
    help="Set a Hail configuration parameter.")
@click.argument('section')
@click.argument('key')
@click.argument('value')
def set(section, key, value):
    validation_func, msg = validations.get(tuple(path), (lambda x: True, ''))
    if not validation_func(value):
        print(f"Error: bad value {value!r} for parameter {key!r} {msg}", file=sys.stderr)
        sys.exit(1)
    if section not in config:
        config[section] = dict()
    config[section][key] = value

    config_file = get_user_config_path()
    with open(config_file, 'w') as f:
        config.write(f)


@config.command(
    help="Unset a Hail configuration parameter (restore to default behavior).")
@click.argument("key")
def unset(section, key):
    if section in config and key in config[section]:
        del config[section][key]
        config_file = get_user_config_path()
        with open(config_file, 'w') as f:
            config.write(f)


@config.command(
    help="Get the value of a Hail configuration parameter.")
@click.argument('section')
@click.argument('key')
def get(section, key):
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
