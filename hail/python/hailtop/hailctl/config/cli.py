import sys
import argparse
import re

from hailtop.config import get_user_config, get_user_config_path

validations = {
    ('batch', 'bucket'): (lambda x: re.fullmatch(r'[^:/\s]+', x) is not None,
                          'should be valid Google Bucket identifier, with no gs:// prefix'),
    ('email',): (lambda x: re.fullmatch(r'.+@.+', x) is not None, 'should be valid email address')
}


def parser():
    main_parser = argparse.ArgumentParser(
        prog='hailctl config',
        description='Manage Hail configuration.')
    subparsers = main_parser.add_subparsers()

    set_parser = subparsers.add_parser(
        'set',
        help='Set a Hail configuration parameter.',
        description='Set a Hail configuration parameter.')
    unset_parser = subparsers.add_parser(
        'unset',
        help='Unset a Hail configuration parameter (restore to default behavior).',
        description='Unset a hail configuration parameter (restore to default behavior).')
    get_parser = subparsers.add_parser(
        'get',
        help='Get the value of a Hail configuration parameter.',
        description='Get the value of a Hail configuration parameter.')
    config_location_parser = subparsers.add_parser(
        'config-location',
        help='Print the location of the config file',
        description='Print the location of the config file')

    set_parser.set_defaults(module='set')
    set_parser.add_argument("parameter", type=str,
                            help="A hail configuration parameter.")
    set_parser.add_argument("value", type=str,
                            help="A value.")

    unset_parser.set_defaults(module='unset')
    unset_parser.add_argument("parameter", type=str,
                              help="A hail configuration parameter.")

    get_parser.set_defaults(module='get')
    get_parser.add_argument("parameter", type=str,
                            help="A hail configuration parameter.")

    config_location_parser.set_defaults(module='config-location')

    return main_parser


def main(args):
    if not args:
        parser().print_help()
        sys.exit(0)
    args = parser().parse_args(args=args)

    config_file = get_user_config_path()
    if args.module == 'config-location':
        print(config_file)
        sys.exit(0)

    config = get_user_config()

    path = args.parameter.split('/')
    if len(path) == 1:
        section = 'global'
        key = path[0]
    elif len(path) == 2:
        section = path[0]
        key = path[1]
    else:
        print('''
Paramters must contain at most one slash separating the configuration section
from the configuration parameter, for example: "batch/billing_project".

Parameters may also have no slashes, indicating the parameter is a global
parameter, for example: "email".

A parameter with more than one slash is invalid, for example:
"batch/billing/project".
'''.lstrip('\n'), file=sys.stderr)
        sys.exit(1)

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
