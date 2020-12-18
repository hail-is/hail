import click

from hailtop.batch_client.client import BatchClient

from .batch import batch
from .batch_cli_utils import get_batch_if_exists


@batch.command(
    help="Cancel a batch.")
@click.argument('batch_id', type=int)
def cancel(batch_id):
    with BatchClient(None) as client:
        maybe_batch = get_batch_if_exists(client, batch_id)
        if maybe_batch is None:
            print(f"Batch with id {batch_id} not found")
            return

        batch = maybe_batch

        batch.cancel()
        print(f"Batch with id {batch_id} was cancelled successfully")
