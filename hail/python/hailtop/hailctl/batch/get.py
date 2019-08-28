import json
import yaml

from .batch_cli_utils import get_batch_if_exists


def init_parser(parser):
    parser.add_argument('batch_id', type=int, help="ID number of the desired batch")
    parser.add_argument('-o', type=str, default='yaml', help="Specify output format",
                        choices=["yaml", "json"])


def main(args, pass_through_args, client):  # pylint: disable=unused-argument
    maybe_batch = get_batch_if_exists(client, args.batch_id)
    if maybe_batch is None:
        print(f"Batch with id {args.batch_id} not found")
        exit(1)

    batch = maybe_batch

    formatter = None
    if args.o == "json":
        def _formatter(s):
            return json.dumps(s, indent=2)
        formatter = _formatter
    elif args.o == "yaml":
        formatter = yaml.dump

    print(formatter(batch.status()))
