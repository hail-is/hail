import sys

import argparse

from . import list_batches
from . import delete
from . import describe
from . import cancel
from . import wait
from . import log


def parser():
    main_parser = argparse.ArgumentParser(
        prog='hailctl batch',
        description='Manage batches running on the batch service managed by the Hail team.'
    subparsers = main_parser.add_subparser()
    
    list_parser = subparsers.add_parser(
        'list',
        help="List batches"
        description="List batches")

def main(args):
    if not args:
        parser().print_help()
        sys.exit(0)
    jmp = {
        'list': list_batches,
        'delete': delete,
        'describe': describe,
        'cancel': cancel,
        'log': log,
        'wait': wait
    }

    args, pass_through_args = parser().parse_known_args(args=args)
    jmp[args.module].main(args, pass_through_args)
