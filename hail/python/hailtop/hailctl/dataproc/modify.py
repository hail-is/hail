import os.path
import sys
import click

from . import gcloud
from .dataproc import dataproc
from .deploy_metadata import get_deploy_metadata


@dataproc.command()
@click.argument('cluster_name')
@click.option('--project',
              metavar='GCP_PROJECT',
              help='Google Cloud project for the cluster.')
@click.option('--zone', '-z',
              metavar='GCP_ZONE',
              help='Compute zone for Dataproc cluster.')
@click.option('--dry-run', is_flag=True,
              help="Print gcloud dataproc command, but don't run it.")
@click.option('--num-workers', '--n-workers', '-w', type=int,
              help="New number of worker machines (min. 2).")
@click.option('--num-secondary-workers', '--num-preemptible-workers', '--n-pre-workers', '-p', type=int,
              help="New number of secondary (preemptible) worker machines.")
@click.option('--graceful-decommission-timeout', '--graceful',
              help='If set, cluster size downgrade will use graceful decommissioning with the given timeout (e.g. "60m").  See `gcloud topic datetimes` for information on duration formats.')
@click.option('--max-idle',
              help='New maximum idle time before shutdown (e.g. "60m").')
@click.option('--no-max-idle', is_flag=True,
              help="Disable auto deletion after idle time.")
@click.option('--expiration-time',
              help=('The time when cluster will be auto-deleted. (e.g. "2020-01-01T20:00:00Z"). '
                    'Execute gcloud topic datatimes for more information.'))
@click.option('--max-age',
              type=str,
              help=('If the cluster is older than this, it will be auto-deleted. (e.g. "2h")'
                    'Execute gcloud topic datatimes for more information.'))
@click.option('--no-max-age', is_flag=True,
              help='Disable auto-deletion due to max age or expiration time.')
@click.option('--update-hail-version', is_flag=True,
              help=("Update the version of hail running on cluster to match "
                    "the currently installed version."))
@click.option('--wheel', help='New Hail installation.')
@click.option('--extra-gcloud-update-args',
              default='',
              help="Extra arguments to pass to 'gcloud dataproc clusters update'")
@click.pass_context
def modify(ctx,
           cluster_name,
           project, zone, dry_run,
           num_workers, num_secondary_workers,
           graceful_decommission_timeout,
           max_idle, no_max_idle,
           expiration_time, max_age, no_max_age,
           update_hail_version, wheel, extra_gcloud_update_args):
    beta = ctx.parent.params['beta']
    print(f'beta {beta}')
    if wheel and update_hail_version:
        print("at most one of --wheel and --update-hail-version allowed", file=sys.stderr)
        sys.exit(1)

    if expiration_time and max_age:
        print("at most one of --expiration-time and --max-age allowed", file=sys.stderr)
        sys.exit(1)

    modify_args = []
    if num_workers is not None:
        modify_args.append('--num-workers={}'.format(num_workers))

    if num_secondary_workers is not None:
        modify_args.append('--num-secondary-workers={}'.format(num_secondary_workers))

    if graceful_decommission_timeout:
        if not modify_args:
            sys.exit("Error: Cannot use --graceful-decommission-timeout without resizing the cluster.")
        modify_args.append('--graceful-decommission-timeout={}'.format(graceful_decommission_timeout))

    if max_idle:
        modify_args.append('--max-idle={}'.format(max_idle))
    if no_max_idle:
        modify_args.append('--no-max-idle')

    if expiration_time:
        modify_args.append('--expiration_time={}'.format(expiration_time))
    if max_age:
        modify_args.append('--max-age={}'.format(max_age))
    if no_max_age:
        modify_args.append('--no-max-age')

    cmd = ['dataproc', 'clusters', 'update', cluster_name] + modify_args

    if beta:
        cmd.insert(0, 'beta')

    cmd.extend(extra_gcloud_update_args.split())

    print("Updating cluster '{}'...".format(cluster_name))
    runner = gcloud.GCloudRunner(project, zone, dry_run)
    runner.run(cmd)

    if update_hail_version:
        deploy_metadata = get_deploy_metadata()
        assert not wheel
        wheel = deploy_metadata["wheel"]

    if wheel is not None:
        wheelfile = os.path.basename(wheel)
        cmds = []
        if wheel.startswith("gs://"):
            cmds.append([
                'compute',
                'ssh',
                '{}-m'.format(cluster_name),
                '--',
                f'sudo gsutil cp {wheel} /tmp/ && '
                'sudo /opt/conda/default/bin/pip uninstall -y hail && '
                f'sudo /opt/conda/default/bin/pip install --no-dependencies /tmp/{wheelfile} && '
                f"unzip /tmp/{wheelfile} && "
                "grep 'Requires-Dist: ' hail*dist-info/METADATA | sed 's/Requires-Dist: //' | sed 's/ (//' | sed 's/)//' | grep -v 'pyspark' | xargs /opt/conda/default/bin/pip install"
            ])
        else:
            cmds.extend([
                [
                    'compute',
                    'scp',
                    wheel,
                    '{}-m:/tmp/'.format(cluster_name)
                ],
                [
                    'compute',
                    'ssh',
                    f'{cluster_name}-m',
                    '--',
                    'sudo /opt/conda/default/bin/pip uninstall -y hail && '
                    f'sudo /opt/conda/default/bin/pip install --no-dependencies /tmp/{wheelfile} && '
                    f"unzip /tmp/{wheelfile} && "
                    "grep 'Requires-Dist: ' hail*dist-info/METADATA | sed 's/Requires-Dist: //' | sed 's/ (//' | sed 's/)//' | grep -v 'pyspark' | xargs /opt/conda/default/bin/pip install"
                ]
            ])

        for cmd in cmds:
            runner.run(cmd)
