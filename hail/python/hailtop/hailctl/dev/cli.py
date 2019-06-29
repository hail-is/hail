import sys

import argparse

from . import deploy


def print_help():
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

    main_parser.print_help()


def main(args):
    if not args:
        print_help()
        sys.exit(0)
    else:
        module = args[0]
        args = args[1:]
        if module == 'benchmark':
            from .benchmark import cli
            cli.main(args)
        elif module == 'deploy':
            from .deploy import cli
            cli.main(args)
        else:
            sys.stderr.write(f"ERROR: no such module: {module!r}")
            print_help()
            sys.exit(1)
