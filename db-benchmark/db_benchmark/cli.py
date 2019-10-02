import sys

import argparse


def print_help():
    main_parser = argparse.ArgumentParser(
        prog='db-benchmark',
        description='Run database benchmarks.')
    subparsers = main_parser.add_subparsers()

    subparsers.add_parser(
        'create-db',
        help='Create a CloudSQL instance.',
        description='Create a CloudSQL instance.')

    subparsers.add_parser(
        'run',
        help='Run benchmarks.',
        description='Run benchmarks.')

    subparsers.add_parser(
        'cleanup-db',
        help='Delete a CloudSQL instance.',
        description='Delete a CloudSQL instance.')

    main_parser.print_help()


def main(args):
    if len(args) < 2:
        print_help()
        sys.exit(0)
    else:
        module = args[1]
        args = args[2:]
        if module == 'create-db':
            from .create_db import create_db
            create_db(args)
        elif module == 'run':
            from .run import run
            run(args)
        elif module == 'cleanup-db':
            from .cleanup_db import cleanup_db
            cleanup_db(args)
        elif module in ('-h', '--help', 'help'):
            print_help()
        else:
            sys.stderr.write(f"ERROR: no such module: {module!r}")
            print_help()
            sys.exit(1)
