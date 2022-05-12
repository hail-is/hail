import os
import sys
import argparse
import re
import warnings

from hailtop.config import get_user_config, get_user_config_path

validations = {
    ('batch', 'bucket'): (lambda x: re.fullmatch(r'[^:/\s]+', x) is not None,
                          'should be valid Google Bucket identifier, with no gs:// prefix'),
    ('batch', 'remote_tmpdir'): (lambda x: any(re.fullmatch(fr'^{scheme}://.*', x) is not None for scheme in ('gs', 's3', 'hail-az')),
                                 'should be valid cloud storage URI such as gs://my-bucket/batch-tmp/'),
    ('email',): (lambda x: re.fullmatch(r'.+@.+', x) is not None, 'should be valid email address'),
}

deprecated_paths = {
    ('batch', 'bucket'): '\'batch/bucket\' has been deprecated. Use \'batch/remote_tmpdir\' instead.'
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
    list_parser = subparsers.add_parser(
        'list',
        help='lists every config variable in the section (default: all sections)',
        description='lists every config variable in the section (default: all sections)')

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

    list_parser.set_defaults(module='list')
    list_parser.add_argument('section', type=str, nargs='?',
                             help='Section to list (default: all sections)')

    return main_parser


def list_config(config, section: str):
    if section:
        for key, value in config.items(section):
            print(f'{key}={value}')
    else:
        for sname, items in config.items():
            for key, value in items.items():
                print(f'{sname}/{key}={value}')


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
    if args.module == 'list':
        list_config(config, args.section)
        sys.exit(0)

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
        path = tuple(path)
        validation_func, msg = validations.get(path, (lambda x: True, ''))
        if not validation_func(args.value):
            print(f"Error: bad value {args.value!r} for parameter {args.parameter!r} {msg}", file=sys.stderr)
            sys.exit(1)
        if path in deprecated_paths:
            warnings.warn(deprecated_paths[path])
        if section not in config:
            config[section] = {}
        config[section][key] = args.value
        try:
            f = open(config_file, 'w', encoding='utf-8')
        except FileNotFoundError:
            os.makedirs(config_file.parent, exist_ok=True)
            f = open(config_file, 'w', encoding='utf-8')
        with f:
            config.write(f)
        sys.exit(0)
    if args.module == 'unset':
        if section in config and key in config[section]:
            del config[section][key]
            with open(config_file, 'w', encoding='utf-8') as f:
                config.write(f)
        sys.exit(0)
    if args.module == 'get':
        if section in config and key in config[section]:
            print(config[section][key])
        sys.exit(0)
    print(f'bad module name: {args.module}')
    sys.exit(1)
