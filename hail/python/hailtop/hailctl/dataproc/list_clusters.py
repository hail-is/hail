import click

from .dataproc import dataproc
from . import gcloud


@dataproc.command(name='list',
                  help="List Dataproc clusters.")
@click.option('--project',
              metavar='GCP_PROJECT',
              help='Google Cloud project for the cluster.')
@click.option('--zone', '-z',
              metavar='GCP_ZONE',
              help='Compute zone for Dataproc cluster.')
@click.option('--dry-run', is_flag=True,
              help="Print gcloud dataproc command, but don't run it.")
@click.option('--extra-gcloud-list-args',
              default='',
              help="Extra arguments to pass to 'gcloud dataproc clusters list'")
def list_clusters(project, zone, dry_run, extra_gcloud_list_args):
    gcloud.GCloudRunner(project, zone, dry_run).run(['dataproc', 'clusters', 'list'] + extra_gcloud_list_args.split())
