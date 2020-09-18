import aiohttp

from ..batch_cli_utils import make_formatter


def init_parser(parser):
    parser.add_argument('billing_project', type=str, help="Name of the desired billing project")
    parser.add_argument('-o', type=str, default='yaml', help="Specify output format",
                        choices=["yaml", "json"])


def main(args, pass_through_args, client):  # pylint: disable=unused-argument
    try:
        billing_project = client.get_billing_project(args.billing_project)
    except aiohttp.client_exceptions.ClientResponseError as cle:
        if cle.code == 403:
            billing_project = None
        raise cle

    print(make_formatter(args.o)(billing_project))
