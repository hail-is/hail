import click

from .dataproc import dataproc


@dataproc.command(
    help="Shut down a Dataproc cluster.")
@click.argument('cluster_name')
@click.option('--async', 'async_', is_flag=True,
              help="Do not wait for cluster deletion.")
@click.option('--extra-gcloud-delete-args',
              default='',
              help="Extra arguments to pass to 'gcloud dataproc clusters delete'")
@click.pass_context
def stop(ctx, cluster_name, *, async_, extra_gcloud_delete_args):
    runner = ctx.parent.obj

    print("Stopping cluster '{}'...".format(cluster_name))

    cmd = ['clusters', 'delete', '--quiet', cluster_name]
    if async_:
        cmd.append('--async')
    cmd.extend(extra_gcloud_delete_args.split())

    runner.run_dataproc_command(cmd)
