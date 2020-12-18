import aiohttp
import click

from hailtop.batch_client.client import BatchClient

from ..batch_cli_utils import make_formatter
from .billing import billing


@billing.command(
    help="Get a particular billing project's info.")
@click.argument('billing_project')
@click.option('--output-format', '-o',
              type=click.Choice(['yaml', 'json']),
              default='yaml', show_default=True,
              help="Specify output format")
def get(billing_project, output_format):
    with BatchClient(None) as client:
        try:
            billing_project = client.get_billing_project(billing_project)
        except aiohttp.client_exceptions.ClientResponseError as cle:
            if cle.code == 403:
                billing_project = None
            raise cle

        print(make_formatter(output_format)(billing_project))
