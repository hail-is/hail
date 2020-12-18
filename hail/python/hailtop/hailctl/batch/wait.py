import sys
from .batch_cli_utils import get_batch_if_exists


def init_parser(parent_subparsers):
    parser = parent_subparsers.add_parser(
        'wait',
        help="Wait for a batch to complete, then print JSON status.",
        description="Wait for a batch to complete, then print JSON status.")
    parser.set_defaults(module='hailctl batch wait')
    parser.add_argument('batch_id', type=int)


def wait(args, client):
    maybe_batch = get_batch_if_exists(client, args.batch_id)
    if maybe_batch is None:
        print(f"Batch with id {args.batch_id} not found")
        sys.exit(1)

    batch = maybe_batch
    print(batch.wait())
