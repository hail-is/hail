import sys
from .batch_cli_utils import get_batch_if_exists, make_formatter


def init_parser(parser):
    parser.add_argument('batch_id', type=int, help="ID number of the desired batch")
    parser.add_argument('-o', type=str, default='yaml', help="Specify output format",
                        choices=["yaml", "json"])


def main(args, pass_through_args, client):  # pylint: disable=unused-argument
    maybe_batch = get_batch_if_exists(client, args.batch_id)
    if maybe_batch is None:
        print(f"Batch with id {args.batch_id} not found")
        sys.exit(1)

    batch = maybe_batch

    print(make_formatter(args.o)(batch.last_known_status()))
