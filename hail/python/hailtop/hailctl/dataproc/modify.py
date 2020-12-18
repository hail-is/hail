import os.path
import sys
import click

from . import gcloud
from .dataproc import dataproc
from .deploy_metadata import get_deploy_metadata


@dataproc.command()
@click.argument('cluster_name')
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
@click.option('--dry-run', is_flag=True,
              help="Print gcloud dataproc command, but don't run it.")
@click.option('--zone', '-z',
              help='Compute zone for Dataproc cluster.')
@click.option('--update-hail-version', is_flag=True,
              help=("Update the version of hail running on cluster to match "
                    "the currently installed version."))
@click.option('--wheel', help='New Hail installation.')
@click.argument('gcloud_args', nargs=-1)
def modify(cluster_name,
           num_workers, num_secondary_workers,
           graceful_decommission_timeout,
           max_idle, no_max_idle,
           expiration_time, max_age, no_max_age,
           dry_run, zone,
           upate_hail_version, wheel, update_hail_version,
           gcloud_args):
    if wheel and update_hail_version:
        print("--wheel and --update-hail-version mutually exclusive", file=sys.stderr)
        sys.exit(1)

    idle_count = int(max_idle) + int(no_max_idle)
    if idle_count != 1:
        print("exactly one of --max-idle and --no-max-idle required", file=sys.stderr)
        sys.exit(1)

    age_count = int(bool(expiration_time)) + int(bool(max_idle)) + int(bool(no_max_idle))
    if age_count != 1:
        print("exactly one of --expiration-time, --max-age, and --no-max-age required", file=sys.stderr)
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
    if expiration_time:
        modify_args.append('--expiration_time={}'.format(expiration_time))
    if max_age:
        modify_args.append('--max-age={}'.format(max_age))

    if modify_args:
        cmd = ['dataproc', 'clusters', 'update', cluster_name] + modify_args

        if beta:
            cmd.insert(0, 'beta')

        if gcloud_args:
            cmd.extend(gcloud_args)

        # print underlying gcloud command
        print('gcloud ' + ' '.join(cmd[:4]) + ' \\\n    ' + ' \\\n    '.join(cmd[4:]))

        # Update cluster
        if not dry_run:
            print("Updating cluster '{}'...".format(cluster_name))
            gcloud.run(cmd)

    if update_hail_version:
        deploy_metadata = get_deploy_metadata()
        assert not wheel
        wheel = deploy_metadata["wheel"]

    if wheel is not None:
        if not zone:
            zone = gcloud.get_config("compute/zone")
        if not zone:
            raise RuntimeError("Could not determine compute zone. Use --zone argument to hailctl, or use `gcloud config set compute/zone <my-zone>` to set a default.")

        wheelfile = os.path.basename(wheel)
        cmds = []
        if wheel.startswith("gs://"):
            cmds.append([
                'compute',
                'ssh',
                '{}-m'.format(cluster_name),
                '--zone={}'.format(zone),
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
                    '--zone={}'.format(zone),
                    wheel,
                    '{}-m:/tmp/'.format(cluster_name)
                ],
                [
                    'compute',
                    'ssh',
                    f'{cluster_name}-m',
                    f'--zone={zone}',
                    '--',
                    'sudo /opt/conda/default/bin/pip uninstall -y hail && '
                    f'sudo /opt/conda/default/bin/pip install --no-dependencies /tmp/{wheelfile} && '
                    f"unzip /tmp/{wheelfile} && "
                    "grep 'Requires-Dist: ' hail*dist-info/METADATA | sed 's/Requires-Dist: //' | sed 's/ (//' | sed 's/)//' | grep -v 'pyspark' | xargs /opt/conda/default/bin/pip install"
                ]
            ])

        for cmd in cmds:
            print('gcloud ' + ' '.join(cmd))
            if not dry_run:
                gcloud.run(cmd)

    if not wheel and not modify_args and gcloud_args:
        sys.stderr.write('ERROR: found pass-through arguments but not known modification args.')
        sys.exit(1)
