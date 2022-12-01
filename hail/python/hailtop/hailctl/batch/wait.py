import json
import sys
from .batch_cli_utils import get_batch_if_exists


def init_parser(parser):
    parser.add_argument('batch_id', type=int)
    parser.add_argument("--quiet", "-q",
                        action="store_true",
                        help="Do not print a progress bar for the batch")
    parser.add_argument('-o', type=str, default='text', choices=['text', 'json'])


def main(args, pass_through_args, client):  # pylint: disable=unused-argument
    maybe_batch = get_batch_if_exists(client, args.batch_id)
    if maybe_batch is None:
        print(f"Batch with id {args.batch_id} not found")
        sys.exit(1)

    batch = maybe_batch
    quiet = args.quiet or args.o != 'text'
    out = batch.wait(disable_progress_bar=quiet)
    if args.o == 'json':
        print(json.dumps(out))
    else:
        print(out)
