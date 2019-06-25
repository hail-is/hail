import aiohttp
from pprint import pprint

def init_parser(parser):
    parser.add_argument('id', type=int)

def main(args, pass_through_args, client):
    batch = None
    try:
        batch = client.get_batch(args.id)
    except aiohttp.client_exceptions.ClientResponseError as cle:
        if cle.code == 404:
            print("Batch with id {} not found".format(args.id))
            return
        raise cle

    pprint(batch.status())
