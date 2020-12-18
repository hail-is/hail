import click

from hailtop.batch_client.client import BatchClient

from ..batch_cli_utils import make_formatter
from .billing import billing

@billing.command(
    name='list',
    help="List billing projects.")
@click.option('--output-format', '-o',
              type=click.Choice(['yaml', 'json']),
              default='yaml', show_default=True,
              help="Specify output format")
def list_billing_projects(output_format):
    with BatchClient(None) as client:
        billing_projects = client.list_billing_projects()
        print(make_formatter(output_format)(billing_projects))
