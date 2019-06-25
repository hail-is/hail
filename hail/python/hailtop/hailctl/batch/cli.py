import sys
import argparse
import aiohttp

from hailtop.batch_client.client import BatchClient
from . import list_batches
from . import delete
from . import describe
from . import cancel
from . import wait
from . import log




def parser():
    main_parser = argparse.ArgumentParser(
        prog='hailctl batch',
        description='Manage batches running on the batch service managed by the Hail team.')
    subparsers = main_parser.add_subparsers()
    
    list_parser = subparsers.add_parser(
        'list',
        help="List batches",
        description="List batches")
    describe_parser = subparsers.add_parser(
        'describe',
        help='Describe a batch',
        description='Describe a batch')
    cancel_parser = subparsers.add_parser(
        'cancel',
        help='Cancel a batch',
        description='Cancel a batch')



    list_parser.set_defaults(module='list')
    
    return main_parser

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
    session = aiohttp.ClientSession(
            raise_for_status=True,
            timeout=aiohttp.ClientTimeout(total=60))
    client = BatchClient(session, url="https://batch.hail.is")

    args, pass_through_args = parser().parse_known_args(args=args)
    jmp[args.module].main(args, pass_through_args, client)
    client.close()
