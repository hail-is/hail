import click

from hailtop.batch_client.client import BatchClient

from .batch import batch
from .batch_cli_utils import get_job_if_exists, make_formatter


@batch.command(
    help="Get the status and specification for a job")
@click.option('batch_id')
@click.option('job_id')
@click.option('--output-format', '-o',
              type=click.Choice(['yaml', 'json']),
              help="Specify output format")
def job(batch_id, job_id, output_format):
    with BatchClient(None) as client:
        maybe_job = get_job_if_exists(client, batch_id, job_id)
        if maybe_job is None:
            print(f"Job with ID {job_id} on batch {batch_id} not found")
            return

        print(make_formatter(output_format)(maybe_job._status))
