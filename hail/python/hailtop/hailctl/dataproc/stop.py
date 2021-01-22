import click

from . import gcloud
from .dataproc import dataproc


@dataproc.command(
    help="Shut down a Dataproc cluster.")
@click.argument('cluster_name')
@click.option('--project',
              metavar='GCP_PROJECT',
              help='Google Cloud project for the cluster.')
@click.option('--zone', '-z',
              metavar='GCP_ZONE',
              help='Compute zone for Dataproc cluster.')
@click.option('--dry-run', is_flag=True,
              help="Print gcloud dataproc command, but don't run it.")
@click.option('--async', 'async_', is_flag=True,
              help="Do not wait for cluster deletion.")
@click.option('--extra-gcloud-delete-args',
              default='',
              help="Extra arguments to pass to 'gcloud dataproc clusters delete'")
def stop(cluster_name, *, project, zone, dry_run, async_, extra_glcoud_delete_args):
    print("Stopping cluster '{}'...".format(cluster_name))

    cmd = ['dataproc', 'clusters', 'delete', '--quiet', cluster_name]
    if async_:
        cmd.append('--async')
    cmd.extend(extra_glcoud_delete_args.split())

    gcloud.GCloudRunner(project, zone, dry_run).run(cmd)
