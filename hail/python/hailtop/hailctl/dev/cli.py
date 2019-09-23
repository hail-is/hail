import sys

import argparse

from . import config
from . import deploy


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
        elif module in ('-h', '--help', 'help'):
            p.print_help()
        else:
            sys.stderr.write(f"ERROR: no such module: {module!r}")
            p.print_help()
            sys.exit(1)
