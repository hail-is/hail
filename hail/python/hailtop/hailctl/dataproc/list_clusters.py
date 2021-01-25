import click

from .dataproc import dataproc


@dataproc.command(name='list',
                  help="List Dataproc clusters.")
@click.option('--extra-gcloud-list-args',
              default='',
              help="Extra arguments to pass to 'gcloud dataproc clusters list'")
@click.pass_context
def list_clusters(ctx, extra_gcloud_list_args):
    runner = ctx.parent.obj
    runner.run_dataproc_command(['clusters', 'list'] + extra_gcloud_list_args.split())
