import sys
import os.path
import click

from .dataproc import dataproc
from .deploy_metadata import get_deploy_metadata


@dataproc.command(
    help="""Modify an existing Dataproc cluster.

'hailctl dataproc modify' works by calling 'gcloud dataproc clusters update' and then updating the Hail version if '--update-hail-version' or '--wheel' is specified.  You can pass arguments to the 'update' command with the option '--extra-gcloud-update-args'.

The following 'gcloud dataproc clusters update' options may be useful:

  --num-workers=NUM_WORKERS: New number of worker machines, minimum 2.

  --num-secondary-workers=NUM_SECONDARY_WORKERS: New number of secondary (preemptible) worker machines.

  --graceful-decommission-timeout=GRACEFUL_DECOMMISSION_TIMEOUT: Graceful decommissioning allows removing nodes from the cluster without interrupting jobs in progress.  Timeout specifies how long to wait for jobs in progress to finish before forcefully removing nodes (and potentially interrupting jobs).  Timeout defaults to 0 if not set (for forceful decommission), and the maximum allowed timeout is 1 day.

  At most one of the following may be set:

    --expiration-time=EXPIRATION_TIME: The time when cluster will be auto-deleted.

    --max-age=MAX_AGE: The lifespan of the cluster before it is auto-deleted, such as '60m' or '1d'.

    --no-max-age: Cancel the cluster auto-deletion by maximum cluster age, as configured by max-age or --expiration-time flags.

  At most one of the following may be set:

      --max-idle=MAX_IDLE: The duration before cluster is auto-deleted after last job finished, such as '60m' or '1d'.

      --no-max-idle: Cancel the cluster auto-deletion by cluster idle duration (configured by --max-idle flag).

  See 'gcloud dataproc clusters update --help' for more information.
""")
@click.argument('cluster_name')
@click.option('--update-hail-version', is_flag=True,
              help=("Update the version of hail running on cluster to match "
                    "the currently installed version."))
@click.option('--wheel', help='New Hail installation.')
@click.option('--extra-gcloud-update-args',
              default='',
              help="Extra arguments to pass to 'gcloud dataproc clusters update'.  The "
              "'update' command is only run if this option is specified.")
@click.pass_context
def modify(ctx,
           cluster_name,
           update_hail_version, wheel, extra_gcloud_update_args):
    runner = ctx.parent.obj

    if wheel and update_hail_version:
        print('at most one of --wheel and --update-hail-version allowed', file=sys.stderr)
        sys.exit(1)

    if not wheel and not update_hail_version and not extra_gcloud_update_args:
        print('nothing to do: none of --wheel, --update-hail-version or --extra-gcloud-update-args specified', file=sys.stderr)
        sys.exit(1)

    if extra_gcloud_update_args:
        print("Updating cluster '{}'...".format(cluster_name))
        cmd = ['clusters', 'update', cluster_name, *extra_gcloud_update_args.split()]
        runner.run_dataproc_command(cmd)

    if update_hail_version:
        deploy_metadata = get_deploy_metadata()
        assert not wheel
        wheel = deploy_metadata["wheel"]

    if wheel is not None:
        wheelfile = os.path.basename(wheel)
        cmds = []
        if wheel.startswith("gs://"):
            cmds.append([
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
                    'scp',
                    wheel,
                    '{}-m:/tmp/'.format(cluster_name)
                ],
                [
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
            runner.run_compute_command(cmd)
