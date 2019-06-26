import json

import yaml

from .batch_cli_utils import get_batch_if_exists

def init_parser(parser):
    parser.add_argument('id', type=int)
    parser.add_argument('-o', type=str, default='yaml')

def main(args, pass_through_args, client):
    maybe_batch = get_batch_if_exists(client, args.id)
    if maybe_batch is None:
        print("Batch with id {} not found".format(args.id))
        return

    batch = maybe_batch

    formatter = None
    if args.o == "json":
        formatter = lambda s: json.dumps(s, indent=2)
    elif args.o == "yaml":
        formatter = yaml.dump

    print(formatter(batch.status()))
