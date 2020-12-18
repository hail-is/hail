import click

from .dataproc import dataproc
from . import gcloud


@dataproc.command(name='list',
                  help="List Dataproc clusters.")
@click.argument('gcloud_args', nargs=-1)
def list_clusters(gcloud_args):
    gcloud.run(['dataproc', 'clusters', 'list'] + list(gcloud_args))
