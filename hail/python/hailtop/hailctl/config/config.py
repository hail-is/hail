import sys
import argparse
import re

from hailtop.config import get_user_config, get_user_config_path

validations = {
    ('batch', 'bucket'): (lambda x: re.fullmatch(r'[^:/\s]+', x) is not None,
                          'should be valid Google Bucket identifier, with no gs:// prefix'),
    ('email',): (lambda x: re.fullmatch(r'.+@.+', x) is not None, 'should be valid email address')
}


def init_parser(parent_subparsers):
    parser = parent_subparsers.add_parser(
        'config',
        help="Manage Hail configuration.",
        description="Manage Hail configuration.")
    subparsers = parser.add_subparsers(
        title='hailctl config subcommand',
        dest='hailctl config subcommand',
        required=True)

    set_parser = subparsers.add_parser(
        'set',
        help="Set a Hail configuration parameter.",
        description="Set a Hail configuration parameter.")
    set_parser.set_defaults(module='hailctl config set')
    set_parser.add_argument("parameter", type=str,
                            help="A hail configuration parameter.")
    set_parser.add_argument("value", type=str,
                            help="A value.")

    unset_parser = subparsers.add_parser(
        'unset',
        help="Unset a Hail configuration parameter (restore to default behavior).",
        description="Unset a hail configuration parameter (restore to default behavior).")
    unset_parser.set_defaults(module='hailctl config unset')
    unset_parser.add_argument("parameter", type=str,
                              help="A hail configuration parameter.")

    get_parser = subparsers.add_parser(
        'get',
        help="Get the value of a Hail configuration parameter.",
        description="Get the value of a Hail configuration parameter.")
    get_parser.set_defaults(module='hailctl config get')
    get_parser.add_argument("parameter", type=str,
                            help="A hail configuration parameter.")

    config_location_parser = subparsers.add_parser(
        'config-location',
        help="Print the location of the config file",
        description="Print the location of the config file")
    config_location_parser.set_defaults(module='hailctl config config-location')

    list_parser = subparsers.add_parser(
        'list',
        help="List every config variable in the section (default: all sections)",
        description="lists every config variable in the section (default: all sections)")
    list_parser.set_defaults(module='hailctl config list')
    list_parser.add_argument('section', type=str, nargs='?',
                             help="Section to list (default: all sections)")


def list_config(config, section: str):
    if section:
        for key, value in config.items(section):
            print(f'{key}={value}')
    else:
        for sname, items in config.items():
            for key, value in items.items():
                print(f'{sname}/{key}={value}')


def main(args):
    config_file = get_user_config_path()
    
    if args.module.startswith('hailctl config config-location'):
        print(config_file)
    elif args.module.startswith('hailctl config list'):
        config = get_user_config()
        list_config(config, args.section)

    if args.module == 'set':
        validation_func, msg = validations.get(tuple(path), (lambda x: True, ''))
        if not validation_func(args.value):
            print(f"Error: bad value {args.value!r} for parameter {args.parameter!r} {msg}", file=sys.stderr)
            sys.exit(1)
        if section not in config:
            config[section] = dict()
        config[section][key] = args.value
        with open(config_file, 'w') as f:
            config.write(f)
        sys.exit(0)
    if args.module == 'unset':
        if section in config and key in config[section]:
            del config[section][key]
            with open(config_file, 'w') as f:
                config.write(f)
        sys.exit(0)
    if args.module == 'get':
        if section in config and key in config[section]:
            print(config[section][key])
        sys.exit(0)
    print(f'bad module name: {args.module}')
    sys.exit(1)
