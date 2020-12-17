import aiohttp

from ..batch_cli_utils import make_formatter


def init_parser(parent_subparsers):
    parser = parent_subparsers.add_parser(
        'get',
        help="Get a particular billing project's info",
        description="Get a particular billing project's info")
    parser.set_defaults(module='hailctl batch billing get')
    parser.add_argument('billing_project', type=str, help="Name of the desired billing project")
    parser.add_argument('-o', type=str, default='yaml', help="Specify output format",
                        choices=["yaml", "json"])


def main(args, client):
    try:
        billing_project = client.get_billing_project(args.billing_project)
    except aiohttp.client_exceptions.ClientResponseError as cle:
        if cle.code == 403:
            billing_project = None
        raise cle

    print(make_formatter(args.o)(billing_project))
