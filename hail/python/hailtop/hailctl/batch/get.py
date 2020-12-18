import sys
import click

from hailtop.batch_client.client import BatchClient

from .batch import batch
from .batch_cli_utils import get_batch_if_exists, make_formatter


@batch.command(
    help="Get a particular batch's info")
@click.argument('batch_id', type=int)
@click.option('--output-format', '-o',
              type=click.Choice(['yaml', 'json']),
              help="Specify output format",)
def get(batch_id, output_format):
    with BatchClient(None) as client:
        maybe_batch = get_batch_if_exists(client, batch_id)
        if maybe_batch is None:
            print(f"Batch with id {batch_id} not found.", file=sys.stderr)
            sys.exit(1)

        batch = maybe_batch

        print(make_formatter(output_format)(batch.last_known_status()))
