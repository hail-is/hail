import sys
import argparse

from hailtop.batch_client.client import BatchClient

from . import list_batches
from . import delete
from . import get
from . import cancel
from . import wait
from . import log
from . import job
from . import billing


def init_parser(parent_subparsers):
    parser = parent_subparsers.add_parser(
        'batch',
        help='Manage batches running on the batch service managed by the Hail team.',
        description='Manage batches running on the batch service managed by the Hail team.')
    subparsers = parser.add_subparsers(
        title='hailctl batch subcommand',
        dest='hailctl batch subcommand',
        required=True)

    list_batches.init_parser(subparsers)
    delete.init_parser(subparsers)
    get.init_parser(subparsers)
    cancel.init_parser(subparsers)
    wait.init_parser(subparsers)
    log.init_parser(subparsers)
    job.init_parser(subparsers)
    billing.init_parser(subparsers)


def main(args):
    client = None
    try:
        client = BatchClient(None)
        if args.module.startswith('hailctl batch billing'):
            billing.main(args, client)
        elif args.module.startswith('hailctl batch list'):
            list_batches.main(args, client)
        elif args.module.startswith('hailctl batch delete'):
            delete.main(args, client)
        elif args.module.startswith('hailctl batch get'):
            get.main(args, client)
        elif args.module.startswith('hailctl batch cancel'):
            cancel.main(args, client)
        elif args.module.startswith('hailctl batch log'):
            log.main(args, client)
        elif args.module.startswith('hailctl batch job'):
            job.main(args, client)
        else:
            assert args.module.startswith('hailctl batch wait')
            wait.main(args, client)
    finally:
        if client:
            client.close()
            client = None
