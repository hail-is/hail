import os.path
import sys

from . import gcloud
from .deploy_metadata import get_deploy_metadata


def init_parser(parser):
    parser.add_argument('name', type=str, help='Cluster name.')
    parser.add_argument('--num-workers', '--n-workers', '-w', type=int,
                        help='New number of worker machines (min. 2).')
    parser.add_argument('--num-secondary-workers', '--num-preemptible-workers', '--n-pre-workers', '-p', type=int,
                        help='New number of secondary (preemptible) worker machines.')
    parser.add_argument('--graceful-decommission-timeout', '--graceful', type=str,
                        help='If set, cluster size downgrade will use graceful decommissioning with the given timeout (e.g. "60m").')
    max_idle_group = parser.add_mutually_exclusive_group()
    max_idle_group.add_argument('--max-idle',
                                type=str,
                                help='New maximum idle time before shutdown (e.g. "60m").')
    max_idle_group.add_argument('--no-max-idle',
                                action='store_true',
                                help='Disable auto deletion after idle time.')
    max_age_group = parser.add_mutually_exclusive_group()
    max_age_group.add_argument(
        '--expiration-time',
        type=str,
        help=('The time when cluster will be auto-deleted. (e.g. "2020-01-01T20:00:00Z"). '
              'Execute gcloud topic datatimes for more information.'))
    max_age_group.add_argument(
        '--max-age',
        type=str,
        help=('If the cluster is older than this, it will be auto-deleted. (e.g. "2h")'
              'Execute gcloud topic datatimes for more information.'))
    max_age_group.add_argument(
        '--no-max-age',
        action='store_true',
        help='Disable auto-deletion due to max age or expiration time.')
    parser.add_argument('--dry-run', action='store_true', help="Print gcloud dataproc command, but don't run it.")
    parser.add_argument('--zone', '-z', type=str, help='Compute zone for Dataproc cluster.')
    wheel_group = parser.add_mutually_exclusive_group()
    wheel_group.add_argument('--update-hail-version', action='store_true', help="Update the version of hail running on cluster to match "
                             "the currently installed version.")
    wheel_group.add_argument('--wheel', type=str, help='New Hail installation.')


async def main(args, pass_through_args):
    modify_args = []
    if args.num_workers is not None:
        modify_args.append('--num-workers={}'.format(args.num_workers))

    if args.num_secondary_workers is not None:
        modify_args.append('--num-secondary-workers={}'.format(args.num_secondary_workers))

    if args.graceful_decommission_timeout:
        if not modify_args:
            sys.exit("Error: Cannot use --graceful-decommission-timeout without resizing the cluster.")
        modify_args.append('--graceful-decommission-timeout={}'.format(args.graceful_decommission_timeout))

    if args.max_idle:
        modify_args.append('--max-idle={}'.format(args.max_idle))
    if args.no_max_idle:
        modify_args.append('--no-max-idle')
    if args.expiration_time:
        modify_args.append('--expiration_time={}'.format(args.expiration_time))
    if args.max_age:
        modify_args.append('--max-age={}'.format(args.max_age))
    if args.no_max_age:
        modify_args.append('--no-max-age')

    if modify_args:
        cmd = ['dataproc', 'clusters', 'update', args.name] + modify_args

        if args.beta:
            cmd.insert(0, 'beta')

        cmd.extend(pass_through_args)

        # print underlying gcloud command
        print('gcloud ' + ' '.join(cmd[:4]) + ' \\\n    ' + ' \\\n    '.join(cmd[4:]))

        # Update cluster
        if not args.dry_run:
            print("Updating cluster '{}'...".format(args.name))
            gcloud.run(cmd)

    wheel = None
    if args.update_hail_version:
        deploy_metadata = get_deploy_metadata()
        wheel = deploy_metadata["wheel"]
    else:
        wheel = args.wheel

    if wheel is not None:
        zone = args.zone if args.zone else gcloud.get_config("compute/zone")
        if not zone:
            raise RuntimeError("Could not determine compute zone. Use --zone argument to hailctl, or use `gcloud config set compute/zone <my-zone>` to set a default.")

        wheelfile = os.path.basename(wheel)
        cmds = []
        if wheel.startswith("gs://"):
            cmds.append([
                'compute',
                'ssh',
                '{}-m'.format(args.name),
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
                    '{}-m:/tmp/'.format(args.name)
                ],
                [
                    'compute',
                    'ssh',
                    f'{args.name}-m',
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
            if not args.dry_run:
                gcloud.run(cmd)

    if not wheel and not modify_args and pass_through_args:
        sys.stderr.write('ERROR: found pass-through arguments but not known modification args.')
        sys.exit(1)
