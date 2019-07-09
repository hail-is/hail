import sys

import argparse


def print_help():
    main_parser = argparse.ArgumentParser(
        prog='hailctl dev',
        description='Manage Hail development utilities.')
    subparsers = main_parser.add_subparsers()

    subparsers.add_parser(
        'benchmark',
        help='Run Hail benchmarks.',
        description='Run Hail benchmarks.')

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
        else:
            sys.stderr.write(f"ERROR: no such module: {module!r}")
            print_help()
            sys.exit(1)
