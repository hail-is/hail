import sys

import asyncio
import argparse

from . import start


def parser():
    main_parser = argparse.ArgumentParser(
        prog='hailctl dataproc',
        description='Manage and monitor Hail HDInsight clusters.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    subparsers = main_parser.add_subparsers()

    start_parser = subparsers.add_parser(
        'start',
        help='Start an HDInsight cluster configured for Hail.',
        description='Start an HDInsight cluster configured for Hail.')

    start_parser.set_defaults(module='start')
    start.init_parser(start_parser)

    return main_parser


def main(args):
    p = parser()
    if not args:
        p.print_help()
        sys.exit(0)
    jmp = {
        'start': start,
    }

    args, pass_through_args = p.parse_known_args(args=args)
    if "module" not in args:
        p.error('positional argument required')

    asyncio.get_event_loop().run_until_complete(
        jmp[args.module].main(args, pass_through_args))
