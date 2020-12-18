import click

from . import gcloud
from .dataproc import dataproc


@dataproc.command(
    help="Shut down a Dataproc cluster.")
@click.argument('cluster_name')
@click.option('--async', 'async_', is_flag=True,
              help="Do not wait for cluster deletion.")
@click.option('--dry-run', is_flag=True,
              help="Print gcloud dataproc command, but don't run it.")
@click.argument('gcloud_args', nargs=-1)
def stop(cluster_name, async_, dry_run, gcloud_args):
    print("Stopping cluster '{}'...".format(cluster_name))

    cmd = ['dataproc', 'clusters', 'delete', '--quiet', cluster_name]
    if async_:
        cmd.append('--async')

    if gcloud_args:
        cmd.extend(gcloud_args)

    # print underlying gcloud command
    print('gcloud ' + ' '.join(cmd[:5]) + ' \\\n    ' + ' \\\n    '.join(cmd[6:]))

    if not dry_run:
        gcloud.run(cmd)
