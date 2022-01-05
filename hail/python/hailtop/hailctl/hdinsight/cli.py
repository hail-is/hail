import sys

import asyncio
import argparse

from . import start
from . import stop
from . import submit
from . import list_clusters


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

    stop_parser = subparsers.add_parser(
        'stop',
        help='Stop an HDInsight cluster configured for Hail.',
        description='Stop an HDInsight cluster configured for Hail.')

    stop_parser.set_defaults(module='stop')
    stop.init_parser(stop_parser)

    submit_parser = subparsers.add_parser(
        'submit',
        help='Submit a job to an HDInsight cluster configured for Hail.',
        description='Submit a job to an HDInsight cluster configured for Hail.')

    submit_parser.set_defaults(module='submit')
    submit.init_parser(submit_parser)

    list_parser = subparsers.add_parser(
        'list',
        help='List HDInsight clusters configured for Hail.',
        description='List HDInsight clusters configured for Hail.')

    list_parser.set_defaults(module='list')
    list_clusters.init_parser(list_parser)

    return main_parser


def main(args):
    p = parser()
    if not args:
        p.print_help()
        sys.exit(0)
    jmp = {
        'start': start,
        'stop': stop,
        'submit': submit,
        'list': list_clusters,
    }

    args, pass_through_args = p.parse_known_args(args=args)
    if "module" not in args:
        p.error('positional argument required')

    asyncio.get_event_loop().run_until_complete(
        jmp[args.module].main(args, pass_through_args))
