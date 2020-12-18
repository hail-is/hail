import sys
import click

from hailtop.batch_client.client import BatchClient

from .batch import batch
from .batch_cli_utils import get_batch_if_exists


@batch.command(
    help="Delete a batch.")
@click.argument('batch_id', type=int)
def delete(batch_id):
    with BatchClient(None) as client:
        maybe_batch = get_batch_if_exists(client, batch_id)
        if maybe_batch is None:
            print(f"Batch with batch_id {batch_id} not found")
            sys.exit(1)

        batch = maybe_batch

        batch.delete()
        print(f"Batch with batch_id {batch_id} was deleted successfully")
