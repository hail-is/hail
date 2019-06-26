import json

import aiohttp
import yaml

def init_parser(parser):
    parser.add_argument('id', type=int)
    parser.add_argument('-o', type=str, default='yaml')

def main(args, pass_through_args, client):
    batch = None
    try:
        batch = client.get_batch(args.id)
    except aiohttp.client_exceptions.ClientResponseError as cle:
        if cle.code == 404:
            print("Batch with id {} not found".format(args.id))
            return
        raise cle

    formatter = None
    if args.o == "json":
        formatter = json.dumps
    elif args.o == "yaml":
        formatter = yaml.dump

    print(formatter(batch.status()))
