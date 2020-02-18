import sys

import argparse

from . import config
from . import deploy
from . import url


def parser():
    main_parser = argparse.ArgumentParser(
        prog='hailctl dev',
        description='Manage Hail development utilities.')
    subparsers = main_parser.add_subparsers()

    config_parser = subparsers.add_parser(
        'config',
        help='Configure deployment',
        description='Configure deployment')

    config.cli.init_parser(config_parser)

    deploy_parser = subparsers.add_parser(
        'deploy',
        help='Deploy a branch',
        description='Deploy a branch')

    deploy.cli.init_parser(deploy_parser)

    url_parser = subparsers.add_parser(
        'url',
        help='Generate a URL for a service',
        description='Generate a URL for a service')

    url.cli.init_parser(url_parser)

    return main_parser


def main(args):
    p = parser()

    if not args:
        p.print_help()
        sys.exit(0)
    else:
        module = args[0]
        if module == 'deploy':
            from .deploy import cli
            args, _ = p.parse_known_args(args=args)
            cli.main(args)
        elif module == 'config':
            from .config import cli
            args, _ = p.parse_known_args(args=args)
            cli.main(args)
        elif module == 'url':
            from .url import cli
            args, _ = p.parse_known_args(args=args)
            cli.main(args)
        elif module in ('-h', '--help', 'help'):
            p.print_help()
        else:
            sys.stderr.write(f"ERROR: no such module: {module!r}")
            p.print_help()
            sys.exit(1)
