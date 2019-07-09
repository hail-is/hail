import sys

import argparse


def print_help():
    main_parser = argparse.ArgumentParser(
        prog='hailctl dev benchmark',
        description='Run and analyze Hail benchmarks.')
    subparsers = main_parser.add_subparsers()

    subparsers.add_parser(
        'run',
        help='Run Hail benchmarks locally.',
        description='Run Hail benchmarks locally.')

    subparsers.add_parser(
        'compare',
        help='Compare Hail benchmarks.',
        description='Run Hail benchmarks.')

    main_parser.print_help()


def main(args):
    if not args:
        print_help()
        sys.exit(0)
    else:
        module = args[0]
        args = args[1:]
        if module == 'run':
            from .run import cli
            cli.main(args)
        elif module == 'compare':
            from .compare import cli
            cli.main(args)
        else:
            sys.stderr.write(f"ERROR: no such module: {module!r}")
            print_help()
            sys.exit(1)
