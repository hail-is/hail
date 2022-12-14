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
from . import submit


def parser():
    main_parser = argparse.ArgumentParser(
        prog='hailctl batch',
        description='Manage batches running on the batch service managed by the Hail team.')
    subparsers = main_parser.add_subparsers()

    billing_parser = subparsers.add_parser(
        'billing',
        help='List billing',
        description='List billing')
    list_parser = subparsers.add_parser(
        'list',
        help="List batches",
        description="List batches")
    get_parser = subparsers.add_parser(
        'get',
        help='Get a particular batch\'s info',
        description='Get a particular batch\'s info')
    cancel_parser = subparsers.add_parser(
        'cancel',
        help='Cancel a batch',
        description='Cancel a batch')
    delete_parser = subparsers.add_parser(
        'delete',
        help='Delete a batch',
        description='Delete a batch'
    )
    submit_parser = subparsers.add_parser(
        'submit',
        help='Submit a batch',
        description='Submit a batch',
    )
    log_parser = subparsers.add_parser(
        'log',
        help='Get log for a job',
        description='Get log for a job'
    )
    job_parser = subparsers.add_parser(
        'job',
        help='Get the status and specification for a job',
        description='Get the status and specification for a job'
    )
    wait_parser = subparsers.add_parser(
        'wait',
        help='Wait for a batch to complete, then print JSON status.',
        description='Wait for a batch to complete, then print JSON status.'
    )

    billing_parser.set_defaults(module='billing')

    list_parser.set_defaults(module='list')
    list_batches.init_parser(list_parser)

    get_parser.set_defaults(module='get')
    get.init_parser(get_parser)

    cancel_parser.set_defaults(module='cancel')
    cancel.init_parser(cancel_parser)

    delete_parser.set_defaults(module='delete')
    delete.init_parser(delete_parser)

    submit_parser.set_defaults(module='submit')
    submit.init_parser(submit_parser)

    log_parser.set_defaults(module='log')
    log.init_parser(log_parser)

    job_parser.set_defaults(module='job')
    job.init_parser(job_parser)

    wait_parser.set_defaults(module='wait')
    wait.init_parser(wait_parser)

    return main_parser


def main(args):
    if not args:
        parser().print_help()
        sys.exit(0)
    jmp = {
        'billing': billing,
        'list': list_batches,
        'delete': delete,
        'get': get,
        'cancel': cancel,
        'log': log,
        'job': job,
        'wait': wait,
        'submit': submit,
    }

    args, pass_through_args = parser().parse_known_args(args=args)

    # hailctl batch doesn't create batches
    client = BatchClient(None)  # type: ignore

    try:
        if args.module == 'billing':
            from .billing import cli  # pylint: disable=import-outside-toplevel
            cli.main(args, pass_through_args, client)
            return

        jmp[args.module].main(args, pass_through_args, client)
    finally:
        client.close()
