import sys

import argparse

from . import deploy


def parser():
    main_parser = argparse.ArgumentParser(
        prog='hailctl dev',
        description='Manage Hail development utilities.')
    subparsers = main_parser.add_subparsers()

    subparsers.add_parser(
        'benchmark',
        help='Run Hail benchmarks.',
        description='Run Hail benchmarks.')

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
        extra_args = args[1:]
        if module == 'benchmark':
            from .benchmark import cli
            cli.main(extra_args)
        elif module == 'deploy':
            from .deploy import cli
            args, pass_through_args = p.parse_known_args(args=args)
            cli.main(args)
        elif module == '-h' or module == '--help' or module == 'help':
            print_help()
        else:
            sys.stderr.write(f"ERROR: no such module: {module!r}")
            p.print_help()
            sys.exit(1)
